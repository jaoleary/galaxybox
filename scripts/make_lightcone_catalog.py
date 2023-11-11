import pandas as pd
import numpy as np
import argparse
from multiprocessing import Pool
import os
import glob
import h5py
from tqdm.auto import tqdm
from scipy.interpolate import interp1d
from astropy import constants as apconst
from astropy import cosmology as apcos
from halotools.mock_observables import radial_distance_and_velocity
from galaxybox.mock_observables import lightcone
from galaxybox.sim_managers.emerge import galaxy_catalog

__author__ = ('Joseph O\'Leary', )


def make_catalog(group):
    files = group[:-2]
    snaparg = group[-2]
    snap_redshift = group[-1]
    
    cat = []
    for i, og in enumerate(vert):
        snap_args = LC.get_snapshots(og)
        if snaparg in snap_args:
            bc = LC.get_boxcoord(og)
            smin, smax = LC.snapshot_extent(snaparg)
            snap_redshift
            # read in data
            galaxies = galaxy_catalog(files).list().copy()
            if len(galaxies) == 0:
                continue
                
            galaxies[['X_pos', 'Y_pos', 'Z_pos']] = LC.transform_position(pos=galaxies[['X_pos', 'Y_pos', 'Z_pos']], randomize=args.randomize, box_coord=bc)
            if args.randomize:
                galaxies[['X_vel', 'Y_vel', 'Z_vel']] = LC.transform_velocity(vel=galaxies[['X_vel', 'Y_vel', 'Z_vel']], box_coord=bc)

            mask = LC.contained(galaxies[['X_pos', 'Y_pos', 'Z_pos']], mask_only=True)
            galaxies = galaxies.loc[mask]
            galaxies[['Redshift', 'Redshift_obs', 'X_cone', 'Y_cone', 'Z_cone', 'R_dist', 'X_cvel', 'Y_cvel', 'Z_cvel', 'R_vel', 'RA', 'Dec']] = pd.DataFrame(columns=['Redshift', 'Redshift_obs', 'X_cone', 'Y_cone', 'Z_cone', 'R_dist', 'X_cvel', 'Y_cvel', 'Z_cvel', 'R_vel', 'RA', 'Dec'])
            galaxies[['X_cone', 'Y_cone', 'Z_cone']] = LC.cone_cartesian(galaxies[['X_pos', 'Y_pos', 'Z_pos']])
            galaxies[['X_cvel', 'Y_cvel', 'Z_cvel']] = LC.cone_cartesian(galaxies[['X_vel', 'Y_vel', 'Z_vel']])

            if method == 'full_width':
                galaxies['R_dist'] = galaxies['Z_cone'].values
                galaxies['R_vel'] = galaxies['Z_cvel'].values
            else:
                xp, yp, zp = galaxies['X_pos'].values, galaxies['Y_pos'].values, galaxies['Z_pos'].values
                xv, yv, zv = galaxies['X_vel'].values, galaxies['Y_vel'].values, galaxies['Z_vel'].values
                xc, yc, zc, vxc, vyc, vzc = 0, 0, 0, 0, 0, 0 # stationary observer at [0,0,0]
                galaxies['R_dist'], galaxies['R_vel'] = radial_distance_and_velocity(xp, yp, zp, xv, yv, zv, xc, yc, zc, vxc, vyc, vzc, np.inf)

            glist = galaxies.copy()  # everything at this snapshot
            # cut out the right radial extent

            mask = (galaxies['R_dist'] >= np.max([smin, D_min])) & (galaxies['R_dist'] < np.min([smax, D_max]))
            galaxies = galaxies.loc[mask]

            # added satellites that break the border
            if args.fuzzy_bounds & (len(galaxies) > 0):
                mains = galaxies.loc[galaxies['Type'] == 0]
                # check if there are any main galaxies
                if len(mains) > 0:
                    mains = mains.loc[(mains['R_dist'].values + mains['Halo_radius'].values) > smax]
                    # do any of these extend over the boundary
                    if len(mains) > 0:
                        sats = glist.loc[glist['Up_ID'].isin(mains.index.values)]
                        mask = (sats['R_dist'] >= smax) & (sats['R_dist'] < D_max)
                        # Add those satellites to the galaxy list.
                        galaxies = pd.concat([galaxies, sats.loc[mask]])

            # Add columnes for lightcone coordinates and apparent redshift
            galaxies[['RA', 'Dec']] = LC.ang_coords(galaxies[['X_pos', 'Y_pos', 'Z_pos']])
            galaxies['Redshift'] = galaxies['R_dist'].apply(redshift_at_D).values
            galaxies['Redshift_obs'] = snap_redshift + galaxies['R_vel'] * (1 + snap_redshift) / apconst.c.to('km/s').value

            cat += [galaxies]

    if len(cat) == 0:
        return None
    else:
        return pd.concat(cat)

def export(data):
    dat = data[0]
    fnum = data[1]
    fp = '.'.join([args.outbase,'{:d}'.format(fnum), 'h5'])
    with h5py.File(fp, 'w') as f:
        data = dat.to_records(index=False)
        dset = f.create_dataset(args.key, data=data, compression='gzip', compression_opts=9)
        # save the info needed to recreate the lightcone object.
        dset.attrs['RA'] = LC.da.value
        dset.attrs['Dec'] = LC.dd.value
        dset.attrs['u1'] = LC.u1
        dset.attrs['u2'] = LC.u2
        dset.attrs['u3'] = LC.u3
        dset.attrs['Lbox'] = LC.Lbox
        dset.attrs['seed'] = LC.seed
        dset.attrs['min_z'] = args.min_z
        dset.attrs['max_z'] = args.max_z
        f.attrs['File number'] = fnum
        f.attrs['N output files'] = args.numfiles

if __name__ == '__main__':
    # setup the command line arguments 
    parser = argparse.ArgumentParser(description='Create mock lightcone catalogs from Emerge/GalaxyNet snapshot data. \
    Lightcones are constructed using the method described in Kitzbichler & White (2007). \
    The light cone geometry will therefor be defined by the angles 1/(m*m*n) x 1/(m*n*n) in radians.')
    parser.add_argument('indir', type=str, help='Input directory containing galaxy catalogs.')
    parser.add_argument('outbase', type=str, help='Output base file name.')
    parser.add_argument('m', type=int, help='Integer as defined in KW07')
    parser.add_argument('n', type=int, help='Integer as defined in KW07')
    parser.add_argument('--min_z', type=float, action='store', default=None, help='Minimum redshift for the galaxy catalog.')
    parser.add_argument('--max_z', type=float, action='store', default=None, help='Maximum redshift for the galaxy catalog.')
    parser.add_argument('--np', type=int, action='store', default=1, help='Number of snapshots to process in parallel.')
    parser.add_argument('--randomize', type=bool, action='store', default=False, help='Randomly shift/rotate galaxies to prevent repetition, default False.')
    parser.add_argument('--fuzzy_bounds', type=bool, action='store', default=True, help='Allow satellites to be collected outside snapshot allowable range, default True.')
    parser.add_argument('--seed', type=int, action='store', default=None, help='Seed used if randomization is enabled.')
    parser.add_argument('--key', type=str, action='store', default='Galaxies', help='Hdf5 dataset key, default `Galaxies`.')
    parser.add_argument('--numfiles', type=int, action='store', default=1, help='Number of output files to split data over')

    method = 'KW07'

    args = parser.parse_args()
    
    files = [name for name in glob.glob(os.path.join(args.indir,'galaxies.*.h5'))]

    redshift = []
    snapnum = []
    for i, fp in enumerate(files):
        with h5py.File(fp, 'r') as f:
            if i == 1: cosmology = apcos.LambdaCDM(H0=f.attrs['HubbleParameter'] * 100,
                                                Om0=f.attrs['Omega_0'],
                                                Ode0=f.attrs['Omega_Lambda'],
                                                Ob0=f.attrs['Omega_Baryon'])
            redshift += [f.attrs['Redshift']]
            snapnum += [f.attrs['SnapNum']]
            Nfiles = f.attrs['NFiles']
            Lbox = f.attrs['BoxSize']
    redshift = np.unique(redshift)
    snapnum = np.unique(snapnum)[::-1]

    if Nfiles > 1:
        file_group = [tuple([os.path.join(args.indir,'galaxies.S{:d}.{:d}.h5'.format(s,j)) for j in range(Nfiles)]+[i]+[redshift[i]]) for i, s in enumerate(snapnum)]
    else:
        file_group = [os.path.join(args.indir,'galaxies.S{:d}.h5'.format(s)) for s in snapnum]

    snap_distance = cosmology.comoving_distance(redshift).value

    # setup the light cone
    LC = lightcone.KW07(m=args.m, n=args.n, Lbox=Lbox)
    LC.set_seed(args.seed)
    LC.set_snap_distances(snap_distance)

    if args.max_z is None:
        args.max_z = redshift[-1]
    if args.min_z is None:
        args.min_z = redshift[0]

    # This will be used to set the redshift of a galaxy based on its radial distance from observer
    z = np.linspace(args.min_z,args.max_z,1000000)
    d = cosmology.comoving_distance(z).value
    redshift_at_D = interp1d(d,z)

    # Check where tesselated volumes should be placed
    D_min = cosmology.comoving_distance(args.min_z).value
    D_max = cosmology.comoving_distance(args.max_z).value
    vert = LC.tesselate(D_min, D_max)

    # set up multiproc if we have it
    if args.np > 1:
        with Pool(args.np) as p:
            catalog = list(tqdm(p.imap(make_catalog, file_group), total=len(file_group),desc="Processing snapshots"))
    else:
        catalog = []
        for fp in tqdm(file_group):
            catalog += [make_catalog(fp)]
    catalog = pd.concat(catalog)
    catalog.index.rename('ID', inplace=True) # this is a safety step incase the first loop turned out no galaxies
    catalog.reset_index(inplace=True)
    catalog.rename(columns={'ID': 'Tree_ID'}, inplace=True)

    split = np.array_split(catalog, args.numfiles)
    out_groups = [tuple([r,i]) for i, r in enumerate(split)]
    # set up multiproc if we have it
    if args.np > 1:
        with Pool(args.np) as p:
            for _ in tqdm(p.imap_unordered(export, out_groups), total=args.numfiles, desc="Distributing data to {:d} files".format(args.numfiles)):
                pass
    else:
        for fp in tqdm(out_groups, desc="Distributing data to {:d} files".format(args.numfiles)):
            export(fp)
