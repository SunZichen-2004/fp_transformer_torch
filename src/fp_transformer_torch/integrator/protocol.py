"""Prescribed healing and fixed-strength production shared by both backends."""

import math


def validate_lambda(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("lambda must be finite and positive")
    return value


def run_fixed_protocol(
    driver,
    state,
    advance,
    production_steps,
    dump_every,
    heal_lambdas,
    heal_steps,
    on_frame,
):
    if int(production_steps) != production_steps or production_steps < 0:
        raise ValueError("production_steps must be a nonnegative integer")
    if int(dump_every) != dump_every or dump_every <= 0:
        raise ValueError("dump_every must be a positive integer")
    healing = [validate_lambda(x) for x in (heal_lambdas or ())]
    if healing and (
        heal_steps is None or int(heal_steps) != heal_steps or heal_steps <= 0
    ):
        raise ValueError(
            "heal_steps must be a positive integer when healing is enabled"
        )
    for i, lam in enumerate(healing):
        driver.lambda_current = lam
        driver.sync_lambda_to_kernel_grad_beta()
        state = advance(state, int(heal_steps))
        if on_frame is not None:
            on_frame("heal", (i + 1) * int(heal_steps), lam, state)
    driver.lambda_current = validate_lambda(driver.production_lambda)
    driver.sync_lambda_to_kernel_grad_beta()
    if on_frame is not None:
        on_frame("production", 0, driver.lambda_current, state)
    done = 0
    while done < production_steps:
        steps = min(int(dump_every), int(production_steps) - done)
        state = advance(state, steps)
        done += steps
        if on_frame is not None:
            on_frame("production", done, driver.lambda_current, state)
    return state
