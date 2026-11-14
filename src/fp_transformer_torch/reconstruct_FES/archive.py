"""Joint-density FES reconstruction from consolidated sFPT / WT-MetaD NPZ files.

Always reconstruct the physical *joint* probability before marginalizing.
All returned free energies use kcal/mol and have their minimum set to zero.
"""

from __future__ import annotations
import itertools, json
import numpy as np
import torch

KB_KCAL = 0.001987204259
EV_TO_KCAL = 23.06054783061903


def _lse(x, axis=None):
    return torch.logsumexp(x, dim=tuple(range(x.ndim)) if axis is None else axis)


def joint_kde(cv, bandwidth, axes, *, periodic=True, device="cpu", chunk=2048):
    """Exact separable evaluation of the source Gaussian kernel, averaged over frames.

    Source distance is mean squared Euclidean distance, or 2-2*mean(cos(delta)).
    Kernel normalization is (2*pi*sigma**2)**(-D/2), matching GaussianKernel.
    Equal replica counts per frame make flattening equivalent to frame averaging.
    """
    cv = np.asarray(cv, dtype=np.float64).reshape(-1, len(axes))
    bw = np.asarray(bandwidth, dtype=np.float64).reshape(-1)
    if (
        len(cv) != len(bw)
        or not len(bw)
        or not np.isfinite(cv).all()
        or not np.isfinite(bw).all()
        or (bw <= 0).any()
    ):
        raise ValueError("paired finite CVs and positive bandwidths are required")
    if len(axes) not in (1, 2, 3):
        raise ValueError("joint KDE supports one to three CVs")
    axes_t = [torch.as_tensor(a, dtype=torch.float64, device=device) for a in axes]
    result = torch.zeros(
        tuple(len(a) for a in axes), dtype=torch.float64, device=device
    )
    d = len(axes)
    for start in range(0, len(bw), chunk):
        c = torch.as_tensor(
            cv[start : start + chunk], dtype=torch.float64, device=device
        )
        b = torch.as_tensor(
            bw[start : start + chunk], dtype=torch.float64, device=device
        )
        factors = []
        for i, axis in enumerate(axes_t):
            delta = axis[:, None] - c[None, :, i]
            exponent = (
                -(1 - torch.cos(delta)) / (d * b[None, :] ** 2)
                if periodic
                else -(delta**2) / (2 * d * b[None, :] ** 2)
            )
            factors.append(torch.exp(exponent))
        weight = (2 * torch.pi * b * b) ** (-d / 2)
        if d == 1:
            result += factors[0] @ weight
        elif d == 2:
            result += (factors[0] * weight) @ factors[1].T
        else:
            result += torch.einsum("ai,bi,ci,i->abc", *factors, weight)
    return (result / len(bw)).cpu().numpy()


def project_log_probability(logp, axes, *, temperature=300.0, reaction=False):
    """Normalize joint probability, then marginalize using log-sum-exp."""
    logp = torch.as_tensor(logp, dtype=torch.float64)
    if not torch.isfinite(logp).all():
        raise ValueError("joint log-probability must be finite")
    logp = logp - _lse(logp)
    kt = KB_KCAL * temperature
    result = {
        "joint_log_probability": logp.numpy(),
        "joint_fes": (-kt * (logp - logp.max())).numpy(),
    }
    for i, a in enumerate(axes):
        result[f"axis_{i}"] = np.asarray(a)
        other = tuple(j for j in range(len(axes)) if j != i)
        marginal = _lse(logp, other) if other else logp
        result[f"fes_1d_{i}"] = (-kt * (marginal - marginal.max())).numpy()
    for i, j in itertools.combinations(range(len(axes)), 2):
        other = tuple(k for k in range(len(axes)) if k not in (i, j))
        marginal = _lse(logp, other) if other else logp
        result[f"fes_2d_{i}_{j}"] = (-kt * (marginal - marginal.max())).numpy()
    mesh = np.meshgrid(*axes, indexing="ij")
    if reaction:
        mask = mesh[1] - mesh[0] < 0
    elif len(axes) == 3:
        mask = np.logical_and.reduce([m < 0 for m in mesh])
    else:
        mask = mesh[0] < 0
    mask = torch.as_tensor(mask)
    if not mask.any() or mask.all():
        raise ValueError("grid does not span both basins")
    result["delta_g_kcal_mol"] = np.asarray(
        float(kt * (_lse(logp[mask]) - _lse(logp[~mask])))
    )
    return result


def reconstruct_sfpt(
    path,
    axes,
    *,
    start_ps=0.0,
    end_ps=None,
    periodic=True,
    temperature=300.0,
    device="cpu",
):
    with np.load(path, allow_pickle=False) as z:
        if not bool(z["complete"]):
            raise ValueError("trajectory archive is incomplete")
        times = z["time_ps"]
        end_ps = (
            float(times[z["phase"] == "production"].max()) if end_ps is None else end_ps
        )
        selected = (z["phase"] == "production") & (times > start_ps) & (times <= end_ps)
        if not selected.any():
            raise ValueError("no frames in requested production window")
        lambdas = z["lambda_value"][selected]
        if not np.all(lambdas == lambdas[0]):
            raise ValueError("FES reconstruction requires fixed production lambda")
        lam = float(lambdas[0])
        density = joint_kde(
            z["cv"][selected],
            z["bandwidth"][selected],
            axes,
            periodic=periodic,
            device=device,
        )
    logp = np.log(np.maximum(density, np.finfo(float).tiny)) * (1 + lam) / lam
    result = project_log_probability(
        logp, axes, temperature=temperature, reaction=not periodic
    )
    result.update(
        density=density,
        production_lambda=np.asarray(lam),
        window_ps=np.asarray([start_ps, end_ps]),
    )
    return result


def reconstruct_metad(
    path, axes, *, temperature=300.0, periodic=True, device="cpu", end_ps=None
):
    from ..integrator.enhanced_sampling import WellTemperedMetadynamicsBias
    from ..math.bspline import periodic_bspline_eval

    with np.load(path, allow_pickle=False) as z:
        if not bool(z["complete"]):
            raise ValueError("trajectory archive is incomplete")
        metadata = json.loads(str(z["metadata_json"]))
        gamma = float(z["bias_factor"])
        if end_ps is None and "bias_coefficients" in z:
            coefficients = torch.as_tensor(z["bias_coefficients"], device=device)
            origin = torch.as_tensor(z["bias_grid_origin"], device=device)
            period = torch.as_tensor(z["bias_grid_period"], device=device)
            mesh = np.meshgrid(*axes, indexing="ij")
            points = torch.as_tensor(
                np.stack(mesh, -1).reshape(-1, len(axes)),
                dtype=coefficients.dtype,
                device=device,
            )
            values = []
            with torch.no_grad():
                for batch in points.split(8192):
                    values.append(
                        periodic_bspline_eval(
                            coefficients, batch, origin=origin, period=period
                        )
                        .cpu()
                        .numpy()
                    )
            potential = np.concatenate(values).reshape(tuple(len(a) for a in axes))
        else:
            # Restore the same B-spline or exact-hill convention used during dynamics.
            count = len(z["hills_heights"])
            if end_ps is not None:
                available = np.flatnonzero(
                    (z["phase"] == "production") & (z["time_ps"] <= end_ps + 1e-10)
                )
                if not len(available):
                    raise ValueError("no metadynamics checkpoint at requested time")
                count = int(z["hills_count"][available[-1]])
            kw = dict(
                n_cv=len(axes),
                sigma=z["hills_sigma"],
                height=float(metadata.get("metad_height", 0.02)),
                bias_factor=gamma,
                temperature=temperature,
                max_hills=0,
                grid_n=int(metadata.get("metad_grid_n", 0)) or None,
                device=device,
                dtype=torch.float32,
            )
            if periodic:
                kw["period"] = 2 * np.pi
            elif kw["grid_n"]:
                kw["grid_min"] = [
                    float(v)
                    for v in str(metadata.get("metad_grid_min", "0,0")).split(",")
                ]
                kw["grid_max"] = [
                    float(v)
                    for v in str(metadata.get("metad_grid_max", "10,10")).split(",")
                ]
            bias = WellTemperedMetadynamicsBias(lambda x: x, **kw)
            bias._ensure_capacity(count)
            bias._centers[:count] = torch.as_tensor(
                z["hills_centers"][:count], device=device
            )
            bias._heights[:count] = torch.as_tensor(
                z["hills_heights"][:count], device=device
            )
            bias.n_hills = count
            bias._rebuild_grid_from_hills()
            points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(
                -1, len(axes)
            )
            potential = np.concatenate(
                [
                    bias.bias_potential_cv(
                        torch.as_tensor(q, dtype=torch.float32, device=device)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    for q in np.array_split(
                        points, max(1, int(np.ceil(len(points) / 8192)))
                    )
                ]
            ).reshape(tuple(len(a) for a in axes))
    fes = -gamma / (gamma - 1) * potential * EV_TO_KCAL
    result = project_log_probability(
        -fes / (KB_KCAL * temperature),
        axes,
        temperature=temperature,
        reaction=not periodic,
    )
    result["bias_factor"] = np.asarray(gamma)
    return result
