import os
import subprocess
import dill


def slurm_or_local(func, iterable, submit_script_env="SLURM_SCRIPT"):
    """
    Run a function over an iterable either locally or via SLURM.

    Parameters:
        func : callable
            The Python function to run on each item.
        iterable : iterable
            The items to iterate over.
        submit_script_env : str
            Name of the environment variable pointing to your SLURM submission script.
    """
    script = os.getenv(submit_script_env)

    for i, item in enumerate(iterable):
        if script:
            # Submit each iteration as a separate SLURM job
            cmd = [
                "sbatch", script,
                "python", "-c",
                f"import dill; func=dill.loads({repr(dill.dumps(func))}); func({repr(item)})"
            ]
            subprocess.run(cmd)
        else:
            # Run locally
            func(item)

