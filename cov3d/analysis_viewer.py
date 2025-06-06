import pickle
import numpy as np
import sys
import os
import tkinter as tk
from tkinter import filedialog, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from cov3d.analyze import create_umap_figure, create_pc_figure

class AnalyzeViewer:
    def __init__(self, master, data,dir):
        self.master = master
        menubar = tk.Menu(self.master)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open", command=self._open_new_file)
        filemenu.add_command(label="Save",command=self._save_cluster_coords)
        menubar.add_cascade(label="File", menu=filemenu)
        self.master.config(menu=menubar)
        self.data = data
        self.dir = dir
        self.coords = data['coords']
        self.umap_coords = data['umap_coords']
        self.cluster_coords = data.get('cluster_coords', None)
        self.umap_cluster_coords = data.get('umap_cluster_coords', None)
        self.selected_cluster_coords = []
        self.figures = {}
        self.color_by = tk.StringVar(value='None')
        self.figure_type = tk.StringVar(value='umap')
        self._setup_gui()
        self._draw_figure()

    def _open_new_file(self):
        path = filedialog.askopenfilename(title='Select analyze_coordinates pkl', filetypes=[('Pickle files', '*.pkl')],initialdir=self.dir)
        if not path:
            return
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.dir = os.path.split(path)[0]
        self.data = data
        self.coords = data['coords']
        self.umap_coords = data['umap_coords']
        self.cluster_coords = data.get('cluster_coords', None)
        self.umap_cluster_coords = data.get('umap_cluster_coords', None)
        self.selected_cluster_coords = []
        self._draw_figure()

    def _setup_gui(self):
        # Top frame for controls (buttons and dropdowns)
        control_frame = ttk.Frame(self.master)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        # Dropdown for figure type
        figure_types = ['umap']
        pc_dim = self.coords.shape[1]
        for i in range(min(5, pc_dim)):
            for j in range(i + 1, min(5, pc_dim)):
                figure_types.append(f'pc_{i}_{j}')
        ttk.Label(control_frame, text='Figure:').pack(side=tk.LEFT, padx=5)
        figure_menu = ttk.OptionMenu(control_frame, self.figure_type, self.figure_type.get(), *figure_types, command=lambda _: self._draw_figure())
        figure_menu.pack(side=tk.LEFT, padx=5)

        # Dropdown for color by
        color_options = ['None','Density'] + [f'PC {i}' for i in range(self.coords.shape[1])]
        ttk.Label(control_frame, text='Color by:').pack(side=tk.LEFT, padx=5)
        color_menu = ttk.OptionMenu(control_frame, self.color_by, self.color_by.get(), *color_options, command=lambda _: self._draw_figure())
        color_menu.pack(side=tk.LEFT, padx=5)

        # Buttons for cluster selection
        ttk.Button(control_frame, text='Reset Cluster Coords', command=self._reset_cluster_coords).pack(side=tk.LEFT, padx=5)

        # Matplotlib figure
        self.fig, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect('button_press_event', self._on_click)


    def _draw_figure(self):
        fig_type = self.figure_type.get()
        color_by = self.color_by.get()
        # Prepare labels for coloring
        if color_by == 'None' or color_by == 'Density':
            labels = None
        else:
            idx = int(color_by.split(' ')[1])
            labels = self.coords[:, idx]

        type_fig = 'hist' if color_by == 'Density' else 'scatter'
        # Use analyze.py plotting functions
        if fig_type == 'umap':
            fig_dict = create_umap_figure(self.umap_coords, self.umap_cluster_coords, labels,fig_type=type_fig)
            fig = fig_dict['umap']
            fig.savefig('test.jpg')
        else:
            i, j = map(int, fig_type.split('_')[1:])
            fig_dict = create_pc_figure(self.coords, self.cluster_coords, labels, num_pcs=max(i, j)+1,fig_type=type_fig)
            fig = fig_dict.get(fig_type)
        # Remove the old canvas and create a new one
        self.canvas.get_tk_widget().pack_forget()
        plt.close(self.fig)  # Close the old figure to avoid memory leaks
        self.fig = fig
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.draw()

    def _on_click(self, event):
        # Only respond to clicks inside the axes
        if event.xdata is None or event.ydata is None:
            return

        fig_type = self.figure_type.get()
        # Get the current visible coordinates
        if fig_type == 'umap':
            coords = self.umap_coords
            cluster_coords = self.umap_cluster_coords
        else:
            i, j = map(int, fig_type.split('_')[1:])
            coords = self.coords[:, [i, j]]
            cluster_coords = self.cluster_coords[:, [i, j]] if self.cluster_coords is not None else None

        # Find the closest point in coords to the click
        click_point = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(coords - click_point, axis=1)
        idx = np.argmin(dists)
        closest_point = coords[idx]


        # vstack to umap_cluster_coords or cluster_coords
        if self.umap_cluster_coords is None:
            self.umap_cluster_coords = np.array([self.umap_coords[idx]])
            self.cluster_coords = np.array([self.coords[idx]])
        else:
            self.umap_cluster_coords = np.vstack([self.umap_cluster_coords, self.umap_coords[idx][None, :]])
            self.cluster_coords = np.vstack([self.cluster_coords,self.coords[idx][None, :]])
            
        self._draw_figure()

    def _reset_cluster_coords(self):
        self.umap_cluster_coords = None
        self.cluster_coords = None
        self._draw_figure()

    def _save_cluster_coords(self):
        path = filedialog.asksaveasfilename(defaultextension='.pkl', filetypes=[('Pickle files', '*.pkl')],initialdir=self.dir,initialfile='latent_coords.pkl')
        if path:
            with open(path, 'wb') as f:
                pickle.dump(self.cluster_coords, f)

    def _on_close(self):
        self.master.quit()
        self.master.destroy()

def main():
    root = tk.Tk()
    root.title('Analyze Coordinates Viewer')
    if(len(sys.argv) >= 2):
        path = sys.argv[1]
    else:
        path = filedialog.askopenfilename(title='Select analyze_coordinates pkl', filetypes=[('Pickle files', '*.pkl')])
        if not path:
            print('No file selected.')
            return
    with open(path, 'rb') as f:
        data = pickle.load(f)
    viewer = AnalyzeViewer(root, data,dir=os.path.split(path)[0])
    root.protocol("WM_DELETE_WINDOW", root.quit)
    root.mainloop()

if __name__ == '__main__':
    main()
