import os
import itertools

DATASET_PATH = "data/scratch_data"

def get_full_path(path_list):

    return [os.path.join(DATASET_PATH,p) if p is not None else None for p in path_list]

def assert_files_exists(files):
    for file in files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f'File {file} not found')

def generate_alg_params(alg_fixed_params, alg_var_params):
    if(isinstance(alg_var_params, list)):
        params_output = [generate_alg_params(alg_fixed_params, p) for p in alg_var_params]
        alg_params,param_description = zip(*params_output)
        return list(itertools.chain.from_iterable(alg_params)),list(itertools.chain.from_iterable(param_description))
    
    alg_params = []
    param_description = []
    keys = alg_var_params.keys()
    alg_var_params = [dict(zip(keys, values)) for values in itertools.product(*alg_var_params.values())]
    for param in alg_var_params:
        alg_params.append({**alg_fixed_params, **param})
        param_description.append(', '.join([f'{k}={v}' for k, v in param.items()]))

    return alg_params,param_description


def run_alg(datasets,run_prefix,params,params_description,run_pipeline=True,run_analysis=True):

    for L, dataset in datasets.items():
        for i, data in enumerate(dataset):
            for param, param_description in zip(params, params_description):
                if(run_pipeline):
                    run_name = f'{run_prefix}_L{L}_{param_description}'
                    alg_param = {**param, 'starfile' : data['dataset'], 'mask' : data['mask'], 'name' : f'"{run_name}"'}
                    command = 'python scripts/comet_pipeline.py ' + ' '.join([f'--{k} {v}' for k, v in alg_param.items()])
                    os.system(command)
                if(run_analysis): #TODO: analysis is run temporaly here, should be part of comet pipeline to log analysis metrics
                    alg_result_dir = os.path.join(os.path.split(data['dataset'])[0],'result_data')
                    command = f'python external/cryobench_analyze.py -i {alg_result_dir} --mask {data["mask"]}'
                    if(data['gt_latent'] is not None):
                        command += f" --gt-latent {data['gt_latent']}"
                    if(data['gt_dir'] is not None):
                        command += f" --gt-dir {data['gt_dir']}"
                    os.system(command)


datasets_L64 = [
    "igg_1d/images/snr0.01/downsample_L64/snr0.01.star",
    "igg_1d/images/snr0.001/downsample_L64/snr0.001.star",
    "igg_rl/images/snr0.01/downsample_L64/snr0.01.star",
    "Ribosembly/images/downsample_L64/snr0.01.star",
    "Spike-MD/images/snr0.1/downsample_L64/particles.star",
    "Tomotwin-100/images/snr0.01/downsample_L64/snr0.01.star"
]

dataset_masks = [
    "igg_1d/init_mask/mask.mrc",
    "igg_1d/init_mask/mask.mrc",
    "igg_rl/init_mask/mask.mrc",
    "Ribosembly/init_mask/mask.mrc",
    "Spike-MD/init_mask/mask.mrc",
    "Tomotwin-100/init_mask/mask.mrc"
]

gt_dir = [
    "igg_1d/vols/128_org",
    "igg_1d/vols/128_org",
    "igg_rl/vols/128_org",
    "Ribosembly/vols/128_org",
    None, #Why is there no GT states for spike md?
    "Tomotwin-100/vols/128_org",
]

gt_latent = [
    "igg_1d/igg_1d_gt_latents.pkl",
    "igg_1d/igg_1d_gt_latents.pkl",
    None,#TODO: figure out igg_rl latent
    None,#TODO: is there GT latent for ribosembly?
    "Spike-MD/gt_latents.pkl",
    None,
]

datasets_L64 = get_full_path(datasets_L64)
dataset_L128 = [dataset.replace("L64", "L128") for dataset in datasets_L64]
dataset_masks = get_full_path(dataset_masks)
gt_dir = get_full_path(gt_dir)
gt_latent = get_full_path(gt_latent)
assert_files_exists(datasets_L64 + dataset_L128 + dataset_masks)
datasets = {#64 : [{'dataset' : dataset, 'mask' : mask,'gt_dir' : gt_dir,'gt_latent' : gt_latent} for dataset, mask, gt_dir, gt_latent in zip(datasets_L64, dataset_masks,gt_dir,gt_latent)],
            128 : [{'dataset' : dataset, 'mask' : mask,'gt_dir' : gt_dir,'gt_latent' : gt_latent} for dataset, mask, gt_dir, gt_latent in zip(dataset_L128, dataset_masks,gt_dir,gt_latent)]}


def reg_scheme_experiment():
    alg_fixed_params = {
        'rank' : 15,
        'lr' : 1e-1,
        'reg' : 1,
        'max-epochs' : 20,
        'batch-size' : 4096,
        'orthogonal-projection' : False,
        'nufft-disc' : 'bilinear',
    }

    alg_var_params = [{
            'use-halfsets' : [False],
            'num-reg-update-iters' : [0,2],
        },
        {
            'use-halfsets' : [True],
            'num-reg-update-iters' : [2],
        }]
    run_prefix = 'test_reg_scheme'

    return alg_fixed_params,alg_var_params,run_prefix    

def pre_cryobench_analyze():
    alg_fixed_params = {
        'rank' : 15,
        'lr' : 1e-1,
        'reg' : 1,
        'max-epochs' : 20,
        'batch-size' : 4096,
        'orthogonal-projection' : False,
        'nufft-disc' : 'bilinear',
        'use-halfsets' : True,
        'num-reg-update-iters' : 2
    }

    alg_var_params = {}
    run_prefix = 'pre_cryobench_analyze'

    return alg_fixed_params,alg_var_params,run_prefix  


alg_fixed_params,alg_var_params,run_prefix = pre_cryobench_analyze()
alg_params_list,alg_param_description = generate_alg_params(alg_fixed_params, alg_var_params)
run_alg(datasets, run_prefix, alg_params_list,alg_param_description,run_pipeline=False)

