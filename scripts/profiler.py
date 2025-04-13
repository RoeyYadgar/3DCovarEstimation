import click
from viztracer import VizTracer
from workflow import covar_workflow,workflow_click_decorator
import torch

@click.command()
@click.option('--profiler-type',type=click.Choice(['viztracer','torch']),default='viztracer',help='Profiler type')
@click.option('--profiler-output',type=str,help = 'Profiler output json file',default='profiler.json')
@workflow_click_decorator
def run_profiling(profiler_type,profiler_output,**workflow_kwargs):
    if(profiler_type == 'viztracer'):
        with VizTracer(output_file=profiler_output,tracer_entries=int(1e6),max_stack_depth=100,min_duration=1e1) as tracer:
            covar_workflow(**workflow_kwargs)
    elif(profiler_type == 'torch'):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
        ) as profiler:
            covar_workflow(**workflow_kwargs)

if __name__ == "__main__":
    run_profiling()