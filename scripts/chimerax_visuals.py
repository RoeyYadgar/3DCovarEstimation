import subprocess
import os
from aspire.volume import Volume
from PIL import Image
from PIL import ImageDraw, ImageFont

CHIMERAX_PATH = '/usr/bin/chimerax'

class CXCFile:
    def __init__(self):
        self.commands = []
        self.view_commands = []
        self.opened_vol = False
        self.num_vol = 0

    def set_view(self,command):
        if(isinstance(command,str)):
            self.view_commands.append(command)
        elif(isinstance(command,list)):
            self.view_commands += command

    def add(self, command: str):
        self.commands.append(command.replace('#i',f'#{self.num_vol}'))
        
        if(command.startswith('open')): #When the first volume is opened save the camera viewing position - reset it whenever a new volume is opened
            if(not self.opened_vol):
                self.commands += self.view_commands
                self.commands.append('view name fixedview')
                self.opened_vol = True
            else:
                self.commands.append('view fixedview')
            self.num_vol += 1
        

    def save(self, file_path: os.path.abspath):
        self.commands.append('exit')
        with open(file_path, "w") as file:
            file.writelines('\n'.join(self.commands))

    def execute(self):
        cxc_path = 'chimerax_commands.cxc'
        self.save(cxc_path)
        #chimerax_command = [CHIMERAX_PATH, "--nogui", "--cmd", f"open {cxc_path}"]
        chimerax_command = [CHIMERAX_PATH, "--cmd", f"open {cxc_path}"]
        print(chimerax_command)
        try:
            subprocess.run(chimerax_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error: {e}")
        os.remove(cxc_path)

CXC = None

def init_CXC():
    global CXC
    CXC = CXCFile()


def save_volume_figure(volume_path,output_image,resolution=500,color='#ffffa0ff',level=None):
    #CXC = CXCFile()
    CXC.add(f'open {volume_path}')
    CXC.add(f'volume #1 color {color}')
    if(level is not None):
        CXC.add(f'volume all level {level}')
    CXC.add('surface dust #1')
    CXC.add(f'save {output_image} transparentBackground true width {resolution} height {resolution} supersample 3')
    CXC.add('close #1')
    #CXC.execute()

def save_volumes_figure(volume_path,output_dir,image_shape = None,vol_prefix='vol',view_commands = [],volume_names = None,remove_individual_figures=True,**chimera_kwargs):

    def prep_kwargs(kwargs_dict,idx):
        out_dict = {}
        for key,val in kwargs_dict.items():
            if(isinstance(val,list)):
                out_dict[key] = val[idx]
            else:
                out_dict[key] = val

        return out_dict

    if(isinstance(volume_path,str)):
        if(os.path.isdir(volume_path)):
            volume_path = [os.path.join(volume_path,v) for v in os.listdir(volume_path) if '.mrc' in v]

    init_CXC()
    CXC.set_view(view_commands)
    if(isinstance(volume_path,str)):
        vols = Volume.load(volume_path)
        num_vols = len(vols)
        
        tmp_vols = []
        outputs = []
        for i in range(num_vols):
            tmp_mrc = f'vol_tmp_{i}.mrc'
            vols[i].save(tmp_mrc,overwrite=True)
            output_vol = os.path.join(output_dir,f'{vol_prefix}_{i}.png')
            save_volume_figure(tmp_mrc,output_vol,**prep_kwargs(chimera_kwargs,i))
            tmp_vols.append(tmp_mrc)
            outputs.append(output_vol)
    elif(isinstance(volume_path,list)):
        outputs = []
        for i,v in enumerate(volume_path):
            output_vol = os.path.join(output_dir,f'{vol_prefix}_{i}.png')
            save_volume_figure(v,output_vol,**prep_kwargs(chimera_kwargs,i))
            outputs.append(output_vol)


    CXC.execute()
    if isinstance(volume_path,str):
        for v in tmp_vols:
            os.remove(v)
            
    if(volume_names is not None):
        for i, output in enumerate(outputs):
            put_text_on_image(output,volume_names[i],output)


    if(image_shape is not None):
        concat_images(outputs,os.path.join(output_dir,f'all_{vol_prefix}.png'),image_shape)

    if(remove_individual_figures):
        for f in outputs:
            os.remove(f)
            
    

def save_eigenvolume_figure(eigenvolume_path,num_eigen,output_dir,image_shape,level,view_commands = [],resolution=500):
    outputs = []
    init_CXC()
    CXC.set_view(view_commands)
    for i in range(num_eigen):
        eigen_pos = eigenvolume_path + f'_pos{i:03d}.mrc'
        eigen_neg = eigenvolume_path + f'_neg{i:03d}.mrc'
        output_vol = os.path.join(output_dir,f'eigenvol_{i}.png')
        #CXC = CXCFile()
        CXC.add(f'open {eigen_pos}')
        CXC.add('volume #1 style surface')
        CXC.add(f'volume #1 color #ff3352')
        CXC.add(f'open {eigen_neg}')
        CXC.add('volume #2 style surface')
        CXC.add(f'volume #2 color #3e5bff')
        CXC.add(f'volume all level {level}')
        #CXC.add(f'turn y 80')
        #CXC.add(f'turn x 100')
        CXC.add(f'save {output_vol} transparentBackground true width {resolution} height {resolution} supersample 3')
        CXC.add('close #1')
        CXC.add('close #2')
        #CXC.execute()
        outputs.append(output_vol)

    CXC.execute()

    concat_images(outputs,os.path.join(output_dir,'all_eigen.png'),image_shape)

def concat_images(images, output_image, output_shape):
    images = [Image.open(x).convert("RGBA") for x in images]
    widths, heights = zip(*(i.size for i in images))

    img_width, img_height = images[0].size
    total_width = img_width * output_shape[1]
    total_height = img_height * output_shape[0]

    new_image = Image.new('RGBA', (total_width, total_height))

    for idx, im in enumerate(images):
        x_offset = (idx % output_shape[1]) * img_width
        y_offset = (idx // output_shape[1]) * img_height
        new_image.paste(im, (x_offset, y_offset), im)

    new_image.save(output_image)


def put_text_on_image(image_path,text,output_path):
    img = Image.open(image_path).convert("RGBA")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", size=60)
    except IOError:
        font = ImageFont.load_default(size=60)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]  
    text_height = bbox[3] - bbox[1] 
    x = (img.width - text_width) // 2
    y = img.height - text_height - 20
    draw.text((x, y), text, fill="black", font=font)
    img.save(output_path)


def discrete_sim():
    view = ['lighting soft intensity -0.5']
    save_volumes_figure('data/rank5_covar_estimate/gt_vols.mrc','data/final_figures/simulation',(2,3),color='#b2ffff',view_commands=view,level=1)
    save_volumes_figure('data/rank5_covar_estimate/gt_vols.mrc','data/final_figures/simulation',(1,6),vol_prefix='gt_vol',color='#b2ffff',view_commands=view,level=1)
    save_volumes_figure('data/rank5_covar_estimate/obj_ml/algorithm_output_1.0/output/analysis/all_volumes','data/final_figures/simulation',(1,6),vol_prefix='reconstructed_snr1',color='#ffffa0ff',view_commands=view,level=1)
    save_volumes_figure('data/rank5_covar_estimate/obj_ml/algorithm_output_0.01/output/analysis/all_volumes','data/final_figures/simulation',(1,6),vol_prefix='reconstructed_snr0.01',color='#ff8a73',view_commands=view,level=0.1)



def igg_1d():
    igg1d_view = ['view orient',
        'zoom 1.5',
        'lighting soft intensity -0.5',
        'turn y 80',
        'turn x 100',
        ]
    save_volumes_figure(['data/scratch_data/igg_1d/images/snr0.01/downsample_L128/result_data/mean_est.mrc'],'data/final_figures/igg_1d/',vol_prefix='mean',view_commands=igg1d_view,remove_individual_figures=False)
    save_eigenvolume_figure('exp_data/igg_analysis/analysis/eigenvol',10,'data/final_figures/igg_1d',(2,5),level=0.002,view_commands=igg1d_view)

    igg1d_view = ['view orient',
            'zoom 2',
            'lighting soft',
            'turn y 80',
            'turn x 100',
            'move x 50'
            ]
    reconstructed_vols = [f'data/scratch_data/igg_1d/images/snr0.01/downsample_L128/result_data/cryobench_output/all_volumes/vol{i:04}.mrc' for i in range(0,100,10)]
    gt_vols = [f'data/scratch_data/igg_1d/vols/128_org/{i:03}.mrc' for i in range(0,100,10)]

    interleaved_vols = []
    levels = []
    colors = []
    k = 10
    for i in range(0, len(reconstructed_vols), k):
        interleaved_vols.extend(reconstructed_vols[i:i+k])
        levels += [0.3]*k
        colors += ['#ffffa0ff']*k
        interleaved_vols.extend(gt_vols[i:i+k])
        levels += [0.003]*k
        colors += ['#b2ffff']*k

    save_volumes_figure(interleaved_vols,'data/final_figures/igg_1d',(2,10),vol_prefix='reconstructed_vol',view_commands=igg1d_view,level=levels,color=colors)

def covar_fsc_simulation():

    micky_view = ['view orient',
            'turn y -90',
            'turn z -90',
            'zoom 0.5'
            ]
    micky_vols  = [os.path.join('data/final_figures/covar_fsc_simulation',v) for v in os.listdir('data/final_figures/covar_fsc_simulation') if '.mrc' in v]

    colors = ['#440154ff',
            '#3b528bff',
            '#21918cff',
            '#5ec962ff',
            '#fde725ff',
            ]
    save_volumes_figure(micky_vols,'data/final_figures/covar_fsc_simulation',(1,5),vol_prefix='micky_vol',
                        view_commands=micky_view,level=0.122,color=colors)
    
def empiar10076():
    empiar_view = ['view orient',
        'lighting soft intensity -0.5',
        'turn z 90',
        'turn x 25',
        ]
    colors = ['#1f77b4',
            '#ff7f0e',
            '#2ca02c',
            '#bcbd22',
            '#008000',
            '#d62728',
            '#ff4500',
            '#800000',
            '#f08080',
            '#9467bd',
            '#6a5acd',
            '#ff00ff',
            '#8a2be2',
            '#dda0dd']
    vol_names = ['A','B','C1','C2','C3','D1','D2','D3','D4','E1','E2','E3','E4','E5']
    dir_name = 'data/scratch_data/empiar10076/downsample_L128/result_data/final_output/analysis/all_volumes'
    vols = [os.path.join(dir_name,f'vol{i:04d}.mrc') for i in range(14)]
    save_volumes_figure(vols,'data/final_figures/empiar10076/',(2,7),view_commands=empiar_view,level=0.014,color=colors,volume_names=vol_names)

    dir_name = 'data/scratch_data/empiar10076/analysis/minor_classes'
    vols = [os.path.join(dir_name,f'vol_{i:03d}.mrc') for i in range(14)]
    save_volumes_figure(vols,'data/final_figures/empiar10076/',(2,7),view_commands=empiar_view,level=0.014,color=colors,volume_names=vol_names,vol_prefix='cryodrgn')
    

    #Same state A from a different angle
    #empiar_view.append('turn y -90')
    #save_volumes_figure(vols[:1],'data/final_figures/empiar10076/',(1,1),vol_prefix='state_A',view_commands=empiar_view,level=0.014,color=colors[:1])

if __name__ == "__main__":
    #discrete_sim()
    #igg_1d()
    #covar_fsc_simulation()
    empiar10076()
