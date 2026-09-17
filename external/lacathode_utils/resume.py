"""LaCATHODE helpers share the framework resume policy."""

from riddle.resume import (
    add_resume_options,
    check_contract,
    inspect_resume,
    record_transition,
    resume_policy,
)

__all__ = ["add_resume_options", "check_contract", "inspect_resume", "record_transition",
           "resume_policy", "contract_settings"]


def contract_settings(runs=None, epochs=None, background="independent"):
    from .pipeline import run_settings

    return run_settings(runs, epochs, background)
