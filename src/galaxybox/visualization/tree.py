from matplotlib import ticker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class TreePlotter():
    tree = None
    pos = None
    node = None
    current_y = 0

    scale = 1
    def __init__(self, min_scale=0, max_mu=10000, mode = 'centered'):
        self.min_scale = min_scale
        self.max_mu = max_mu
        self.mode = mode

    def __call__(self, x, ax=None):
        self._init_tree(x)
        if ax is None:
            self._init_ax()
        else:
            self.ax=ax
        
        self.set_pos()
        self.render()
        self._reset()

    def _reset(self):
        self.tree = None
        self.pos = None
        self.node = None
        self.current_y = 0

    def _init_tree(self, x):
        self.tree = x

        self.pos = pd.DataFrame(index=self.tree.index.values, columns=['x','y'])
        self.pos.index.name = 'id'
        self.pos['x'] = self.tree.scale.values

        self.node = self.tree.index[0]

    def set_pos(self):
        # set the position of current node
        self.pos.loc[self.node, 'y'] = self.current_y
        self.scale = self.tree.loc[self.node].scale
        #self.pos.loc[self.node, 'y'] = self.ypos[self.scale]

        # get the ID of the most massive progenitor
        immp = 0
        if self.tree.loc[self.node]['num_prog'] > 0:
            immp = self.tree.loc[(self.tree.desc_id == self.node) & (self.tree.mmp == 1)].index.values[0]

        # doing a mass ratio check on coprogs
        icoprog = int(self.tree.loc[self.node]['Next_coprogenitor_depthfirst_ID'])
        if icoprog > 0:
            icoprog = self.tree.loc[self.tree.Depth_first_ID == icoprog].index.values[0]
            dat_c = self.tree.loc[icoprog]
            did_c = int(dat_c.desc_id)
            main_prog = self.tree.loc[(self.tree.desc_id == did_c) & (self.tree.mmp == 1)]
            mu = 10**(float(main_prog.mvir) - dat_c.mvir)    
            if mu > self.max_mu: icoprog = 0

        
        if self.scale > self.min_scale:
            # walk the main branch first
            if immp > 0:
                self.node = immp
                self.set_pos()
            # walk the coprogenitors
            if icoprog > 0:
                self.current_y += 1
                #self.ypos[self.scale] += 1
                self.node = icoprog
                self.set_pos()

            if self.mode == 'centered':
                self._update_desc_pos()

    def _get_mmp_id(self, idx):
        if self.tree.loc[idx]['num_prog'] > 0:
            return  self.tree.loc[(self.tree.desc_id == idx) & (self.tree.mmp == 1)].index.values[0]

    def _update_desc_pos(self):
        desc_id = self.tree.loc[self.node].desc_id
        max_depth = 1
        depth = 0

        while (desc_id > 0) and (depth < max_depth):
            if self.tree.loc[self.node].mmp == 0:

                immp = self._get_mmp_id(desc_id)
                mmp_y = self.pos.loc[immp, 'y']
                now_y = self.pos.loc[self.node, 'y']

                self.pos.loc[desc_id, 'y'] = (now_y+mmp_y)/2
                #self.pos.loc[desc_id, 'y'] += 0.5
                depth += 1
            else:
                self.pos.loc[desc_id, 'y'] = self.pos.loc[self.node, 'y']
            
            self.node = desc_id
            desc_id = self.tree.loc[self.node].desc_id

    def render(self):

        # some indices are blank since they didnt meet mu or scale cuts
        # so we need to ignore those
        valid_idx = ~self.pos['y'].isna()
        for idx in self.pos[valid_idx].index.values:
            if self.tree.loc[idx].desc_id == 0: continue

            desc_id = self.tree.loc[idx].desc_id
            x = [self.pos.loc[idx]['x'], self.pos.loc[desc_id]['x']]
            y = [self.pos.loc[idx]['y'], self.pos.loc[desc_id]['y']]
            self.ax.plot(x,y,color='k',zorder=0)
        
        self.ax.scatter(
            self.pos.loc[valid_idx]['x'],
            self.pos.loc[valid_idx]['y'],
            c=self.tree.loc[valid_idx]['mvir'],
            zorder=10,
            vmin=9,
            vmax = self.tree.mvir.max(),
            cmap = plt.cm.jet
        )
    
    def _init_ax(self):
        vmin = 9
        vmax = self.tree.mvir.max()
        self.fig, self.ax = plt.subplots(figsize=(20,10))
        self.ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0,1.1,0.1)))
        self.ax.xaxis.set_minor_locator(ticker.FixedLocator(np.arange(0,1,0.025)))
        self.ax.set_xlabel('Scale factor', fontsize=18)
        self.ax.tick_params(axis='y',  labelleft=False)
        self.ax.set_yticks([])

        sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        self.cbar = plt.colorbar(sm, pad=0.01, aspect=30)
        self.cbar.ax.xaxis.set_label_position('top')
        self.cbar.ax.xaxis.set_ticks_position('top')
        self.cbar.ax.minorticks_on()
        self.cbar.set_label('$\log_{10}(M_{\\mathrm{h}}/M_{\odot})$', fontsize=18)
