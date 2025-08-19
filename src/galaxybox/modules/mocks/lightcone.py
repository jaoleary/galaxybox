"""Class for constucting light cones."""

from typing import Optional, Union

import astropy.units as apunits
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import Angle
from scipy.spatial.transform.rotation import Rotation
from shapely.geometry import Polygon

from galaxybox.utils.functions import (
    coordinate_plane,
    poly_traverse,
    rotate,
    shuffle_string,
    translate,
)
from galaxybox.visualization.plot import Arrow3D, render_cube


class Lightcone:
    """Class representing a lightcone object.

    This class is used to initialize and manipulate a lightcone object, which represents a
    cone-shaped region of the universe. It provides methods for creating different types of
    lightcones and performing operations on them.

    """

    def __init__(self, da, dd, u1, u2, u3, lbox, full_width=False):
        """Initialize a lightcone object.

        Parameters
        ----------
        da : Astropy Angle
            Right ascension of lightcone .
        dd : Astropy Angle
            Declination of lightcone.
        u1 : 1-D array
            a vector with shape (1,3) aligned with the right ascension.
        u2 : 1-D array
            a vector with shape (1,3) aligned with the declination.
        u3: 1-D array
            A vector with shape(1,3) aligned with line of sight
        lbox : float
            Comoving cosmological box side length.
        full_width : Bool
            If True, removes angle checks and sets distances only along los axis only

        """
        self.da, self.dd, self.u1, self.u2, self.u3, self.lbox, self.full_width = (
            da,
            dd,
            u1,
            u2,
            u3,
            lbox,
            full_width,
        )
        if not self.full_width:
            corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
            corners[:, 0] = corners[:, 0] * da.value / 2
            corners[:, 1] = corners[:, 1] * dd.value / 2
            self.plane = Polygon(corners)
            dcm = np.matmul(np.vstack([u1, u2, u3]), np.identity(3))
            self.rotation = Rotation.from_matrix(dcm)

        # TODO: I dont like how this sets the initial seed....
        self.seed = np.random.randint(2**12)

    @classmethod
    def kw07(cls, m, n, lbox):
        """Create a lightcone using the method of Kitzblicher and White 2007.

        Parameters
        ----------
        m : int
            Description of parameter `m`.
        n : int
            Description of parameter `n`.
        lbox : float
            Comoving cosmological box side length.

        Returns
        -------
        lightcone
            A lightcone object.

        """
        # create lightcone coordinate vectors
        u3 = np.array([n, m, m * n]) / np.sqrt(m**2 + n**2 + (m * n) ** 2)
        # construct univector corresponding to the smaller axis m (x-axis) and n (y-axis).
        if m <= n:
            uax = np.array([1, 0, 0])
        else:
            uax = np.array([0, 1, 0])
        u1 = np.cross(u3, uax)
        u1 /= np.linalg.norm(u1, axis=-1)
        u2 = np.cross(u3, u1)

        # light cone angles
        da = Angle(1 / (m * m * n), apunits.radian)
        dd = Angle(1 / (m * n * n), apunits.radian)

        return cls(da, dd, u1, u2, u3, lbox)

    @classmethod
    def full_width(cls, lbox, los_axis=2):
        """Create a 'lightcone' with rectangle geometry where the projected side lengths are lbox.

        Parameters
        ----------
        lbox : float
            Comoving cosmological box side length.
        los_axis : int, optional
            The axis to be used as the line of sight, by default 2

        Returns
        -------
        lightcone
            A lightcone object.

        """
        u3 = np.zeros(3)
        u3[los_axis] = 1
        u1 = np.roll(u3, 1)
        u2 = np.roll(u1, 1)
        da = Angle(np.pi, apunits.radian)
        dd = Angle(np.pi, apunits.radian)
        return cls(da, dd, u1, u2, u3, lbox, full_width=True)

    @classmethod
    def hybrid(cls, ra, dec, lbox):
        """Create a lightcone using the method of Kitzblicher and White 2007.

        This method requires specifying Right ascension and declination directly

        Parameters
        ----------
        ra : int, float, string
            Right ascension of lightcone in radians.
        dec : int, float, stri
            Declination of lightcone in radians.
        lbox : float
            Comoving cosmological box side length.

        Returns
        -------
        lightcone
            A lightcone object.

        """
        if isinstance(ra, (float, int, str)):
            ra = Angle(ra, apunits.radian)
        if isinstance(dec, (float, int, str)):
            dec = Angle(dec, apunits.radian)

        n = (ra.value / (dec.value**2)) ** (1 / 3)
        m = n * dec.value / ra.value
        if (m < 1) or (n < 1):
            raise ValueError("Selected angles are too wide, choose small aperture")
        # create lightcone coordinate vectors
        u3 = np.array([n, m, m * n]) / np.sqrt(m**2 + n**2 + (m * n) ** 2)

        if m <= n:
            uax = np.array([1, 0, 0])
        else:
            uax = np.array([0, 1, 0])
        u1 = np.cross(u3, uax)
        u1 /= np.linalg.norm(u1, axis=-1)
        u2 = np.cross(u3, u1)

        # light cone angles
        da = Angle(1 / (m * m * n), apunits.radian)
        dd = Angle(1 / (m * n * n), apunits.radian)
        return cls(da, dd, u1, u2, u3, lbox)

    def vector(self, d=1):
        """Return a vector of length D along the light cone line of sight."""
        return d * self.u3

    def contained(
        self, pos: np.ndarray, d_min: float = 0, d_max: float = np.inf, mask_only: bool = False
    ) -> Union[np.ndarray, bool]:
        """Check if a set of positions are contained within the lightcone.

        Parameters
        ----------
        pos : np.ndarray
            An array of size (N, 3) containing the 3D cartesian positions for N points in space.
        d_min : float, optional
            The minimum comoving distance to consider, by default 0.
        d_max : float, optional
            The maximum comoving distance to consider, by default np.inf.
        mask_only : bool, optional
            If True, return a boolean mask indicating which positions are contained, by default
            False.

        Returns
        -------
        Union[np.ndarray, bool]
            If mask_only is False, returns a boolean array indicating which positions are contained.
            If mask_only is True, returns a boolean mask indicating which positions are contained.

        """
        pos = np.atleast_2d(pos).astype(np.float64)

        if self.full_width:
            los_arg = np.argmax(self.u3)
            mask = (pos[:, los_arg] >= d_min) & (pos[:, los_arg] < d_max)
        else:
            tan_a = np.inner(pos, self.u1) / np.inner(pos, self.u3)
            tan_d = np.inner(pos, self.u2) / np.inner(pos, self.u3)
            mask_alpha = np.abs(tan_a) <= np.tan(self.da / 2)
            mask_delta = np.abs(tan_d) <= np.tan(self.dd / 2)
            mask_distance = (np.sqrt(((pos) ** 2).sum(axis=1)) >= d_min) & (
                np.sqrt(((pos) ** 2).sum(axis=1)) < d_max
            )
            mask = mask_alpha & mask_delta & mask_distance

        if mask_only:
            return mask
        else:
            return pos[mask]

    def snapshot_extent(self, snapnum):
        """Return minimum and maximum radial extent of a snapshot from observer origin."""
        if not hasattr(self, "distance"):
            raise NameError("  'distance' not defined, set using `set_snap_distances`.")

        if snapnum == 0:
            min_d = 0
            max_d = self.distance[snapnum + 1] / 2
        else:
            min_d = (self.distance[snapnum] + self.distance[snapnum - 1]) / 2
            max_d = (self.distance[snapnum] + self.distance[snapnum + 1]) / 2

        return min_d, max_d

    def ang_coords(self, pos):
        """Return the 2D angular coordinates of a point in 3D space.

        Projects a 3D point space onto the line of sight plane of the light cone.

        Parameters
        ----------
        pos : 2-D array
            An array of size (N, 3) containing the 3D cartesian positions for N
            points in space.

        Returns
        -------
        angular_coordinates : array_like
            An array of size (N, 2) containing the anglular coordinates of each point.

        """
        pos = np.atleast_2d(pos)
        alpha = np.arctan2(np.inner(pos, self.u1), np.inner(pos, self.u3))
        delta = np.arctan2(np.inner(pos, self.u2), np.inner(pos, self.u3))
        return np.array([alpha, delta]).T

    def cone_cartesian(self, pos):
        """Convert 3-D carteisian coordinates to cone...space cartesian coordinates.

        Parameters
        ----------
        pos : 2-D array
            An array of size (N, 3) containing the 3D cartesian positions for N
            points in space.

        Returns
        -------
        pos : 2-D array
            Positions with respect to cone coordinate system.

        """
        pos = np.atleast_2d(pos)
        return np.array([np.inner(pos, self.u1), np.inner(pos, self.u2), np.inner(pos, self.u3)]).T

    def plane_intersect(self, vertices):
        """Check if a plane in 3D space will intesect a lightcone in projection.

        An plan defined in 3D space by a list of vertex coordinates is projected
        into the observing plane of the lightcone. A polygon is constructed
        according to the projected vertex coordinates. A check is then performed
        to see if the polygon formed by the plane in projection will overlap with
        the polygon formed by the observing area of the light cone.

        Parameters
        ----------
        vertices : 2-D array
            An array of size (N,3) containing the 3D coordinates of some plane geometry.

        Returns
        -------
        bool
            A True/False values indicating whether the planes overlap.

        """
        plane_angcoords = poly_traverse(self.ang_coords(vertices))
        poly = Polygon(plane_angcoords)
        return self.plane.intersects(poly)

    def tesselate(self, d_min, d_max):
        """Determine where additional volumes should be placed to fill lightcone.

        Parameters
        ----------
        d_min : float
            Minimum comoving distance to place new volumes
        d_max : float
            Maximum comoving distance to place new volumes

        Returns
        -------
        origins : 2-D array
            An array containing the 3D coordinates where additional volumes should be placed.

        """
        # TODO: This method needs a performance boost

        if self.full_width:
            tess_max = np.ceil(d_max / self.lbox).astype(int)
            origins = np.zeros([tess_max, 3])
            origins[:, np.argmax(self.u3)] = np.arange(0, tess_max) * self.lbox
            mask = origins[:, np.argmax(self.u3)] + self.lbox >= d_min
            return origins[mask]

        # Find the maximum number of tesselations we might need in any direction
        # along the line of sight.
        tess_max = np.max(np.ceil(self.vector(d_max) / self.lbox).astype(int)) + 1
        # now check the corners of the light cone.
        r = [-0.5, 0.5]
        for i, r1 in enumerate(r):
            for j, r2 in enumerate(r):
                vec = rotate(self.u3, angle=self.dd.value * r1, u=self.u1)
                vec = rotate(vec, angle=self.da.value * r2, u=self.u2)
                tess_max = max(tess_max, np.max(np.ceil(vec * d_max / self.lbox).astype(int)) + 1)

        origins = np.zeros([tess_max**3, 3])
        mask = np.full(len(origins), False)

        planes = np.array([[0, 1], [0, 2], [1, 2]])
        c = 0
        h = np.sqrt(3 * self.lbox**2)

        # Generate all indices for the 3D grid
        indices = np.ndindex((tess_max, tess_max, tess_max))

        # Iterate over the indices
        for c, (i, j, k) in enumerate(indices):
            origins[c, :] = [i * self.lbox, j * self.lbox, k * self.lbox]
            rad = np.linalg.norm(origins[c, :])
            if (rad + h >= d_min) & (rad < d_max):
                for p in planes:
                    # here we check if any plane of the simulation volume intersects with
                    # the light cone in projection this is method is a bit expensive, there
                    # are certainly ways to speed this up.
                    if self.plane_intersect(
                        coordinate_plane(origin=origins[c, :], lbox=self.lbox, axes=p)
                    ):
                        mask[c] = True
                        break
        return origins[mask]

    def los_velocity(self, vec):
        """Return velocity along light cone line of sight.

        Parameters
        ----------
        vec : array_like
            An array of velocity vectors with shape (N,3)

        Returns
        -------
        los_vel : float
            The velocity along the line of sight.

        """
        return np.inner(vec, self.u3)

    def set_snap_distances(self, distances):
        """Set the comoving distance to each simulation snapshot.

        Parameters
        ----------
        distances : 1-D array of length N-snapshots
            An ascending array containing the comoving distance to each snapshot based on the
            snapshot output redshift.

        """
        setattr(self, "distance", distances)

    def get_snapshots(self, origin):
        """Detemine which snap should could my applied to a tesselate volume.

        Snap shots used is based on its comoving distance from the observer

        Parameters
        ----------
        origin : 1-D array
            The 3D coordinates specifying the origin location of a tesselate box.

        Returns
        -------
        snap_arg : 1-D array
            An array with the arguments of applicable snapshots based on the `distances` attribute.

        """
        # get the minimum and maximum extent of a box wrt to the origin
        dmin = np.linalg.norm(origin)
        dmax = np.linalg.norm(origin + self.lbox)

        snap_max_d = np.concatenate(
            (self.distance[:-1] + np.diff(self.distance) / 2, [self.distance[-1]])
        )
        snap_min_d = np.concatenate(
            ([self.distance[0]], self.distance[1:] - np.diff(self.distance) / 2)
        )
        # which snapshots fall in that distance range.
        far = np.logical_and(snap_max_d >= dmin, snap_max_d < dmax)
        near = np.logical_and(snap_min_d >= dmin, snap_min_d < dmax)
        snap_arg = np.argwhere(near | far).T[0]

        return snap_arg

    def set_seed(self, seed=None):
        """Set the random seed for this cone.

        Parameters
        ----------
        seed : int, optional
            The seed to be used for randomizing box transformations, by default None

        """
        self.seed = seed

    def random_angles(self, length: int = 1, box_coord: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate a list of random 3D axis rotation angles.

        Parameters
        ----------
        length : int, optional
            The number of rotation angles requested, by default 1
        box_coord : np.ndarray, optional
            The sub-box coordinate for each rotation angle, by default None

        Returns
        -------
        np.ndarray
            An array of rotation angles with shape (length, 3)

        """
        box_seed = "".join((box_coord).astype(int).astype(str))
        if self.seed is not None:
            np.random.seed(int(str(self.seed) + box_seed))
        angle = np.random.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2], (length, 3))
        sequence = np.array([shuffle_string("xyz") for _ in range(length)])
        return angle, sequence

    def random_translations(self, length=1, box_coord=None):
        """Return a list of random 3D translation lengths.

        Parameters
        ----------
        length : int, optional
            The number of rotation angles requested, by default 1
        box_coord : np.ndarray (N, 3), optional
            The sub-box coordinate for each translation length, by default None

        Returns
        -------
        np.ndarray (length, 3)
            An array of translations of specified length

        """
        box_seed = "".join((box_coord).astype(int).astype(str))
        if self.seed is not None:
            np.random.seed(int(str(self.seed) + box_seed))
        return np.random.uniform(low=0, high=self.lbox, size=(length, 3))

    def transform_position(self, pos, box_coord, randomize=False):
        """Adjust coordinates based on the randomization within a sub-box located at `box_coord`.

        Parameters
        ----------
        pos : np.ndarray (N, 3)
            An array of caretesian coordinates
        box_coord : np.ndarray (N, 3)
            The sub-box coordinate for each input postion.
        randomize : bool, optional
            Should these points have sub-box random rotation and translations applied, by default
            True

        Returns
        -------
        pos : np.ndarray (N, 3)
            An updated array of caretesian coordinates

        """
        pos = np.atleast_2d(pos)
        box_coord = np.atleast_1d(box_coord).astype(int)

        # set the new origin for these points
        new_origin = box_coord * self.lbox

        # get the randomization parameters for this subbox
        angles, sequences = self.random_angles(box_coord=box_coord)
        translations = self.random_translations(box_coord=box_coord)

        if randomize:
            rot = Rotation.from_euler(seq=sequences[0], angles=angles[0])

            # apply coordinate rotation
            pos = rot.apply(pos)
            if pos[:, 0].min() < 0:
                pos[:, 0] += self.lbox
            if pos[:, 1].min() < 0:
                pos[:, 1] += self.lbox
            if pos[:, 2].min() < 0:
                pos[:, 2] += self.lbox

            # apply translations
            pos = translate(pos, lbox=self.lbox, axes=[0, 1, 2], dx=translations[0])

        # move the origin for these points
        pos = pos + new_origin

        return pos

    def transform_velocity(self, vel, box_coord=None):
        """Adjust velocity based on the randomization within a sub-box located at `box_coord`.

        Parameters
        ----------
        vel : np.ndarray (N, 3)
            An array of velocties
        box_coord : np.ndarray (N, 3)
            The sub-box coordinate for each input postion.

        Returns
        -------
        vel : np.ndarray (N, 3)
            An updated array of velocties

        """
        vel = np.atleast_2d(vel)
        box_coord = np.atleast_1d(box_coord).astype(int)
        angles, sequences = self.random_angles(box_coord=box_coord)

        rot = Rotation.from_euler(seq=sequences[0], angles=angles[0])

        vel = rot.apply(vel)
        return vel

    def get_boxcoord(self, pos):
        """Find sub-box coordinate at a given position.

        Parameters
        ----------
        pos : np.ndarray (N, 3)
            An array of cartesian coordinates.

        Returns
        -------
        box_coord : np.ndarray (N, 3)
            The sub-box coordinate for each input postion.

        """
        box_coord = np.floor(pos / self.lbox).astype(int)
        return box_coord

    def plot_cone(
        self,
        d_min,
        d_max,
        equal_aspect=True,
        los=True,
        tesselations=True,
        cone_edges=False,
    ):
        """Visualize lightcone geometry and box tesselations.

        Parameters
        ----------
        d_min : float
            Minimum comoving distance to place new volumes
        d_max : float
            Maximum comoving distance to place new volumes
        equal_aspect : bool, optional
            Specifies whether plotting axis will have equal lengths, by default True
        los : bool, optional
            Specifies whether to plot the line of sight arrow, by default True
        tesselations : bool, optional
            Specifies whether to plot the box tesselations, by default True
        cone_edges : bool, optional
            Specifies whether to plot the cone edges, by default False

        """
        # TODO: This method is a complete mess. It works but needs serious cleanup

        vert = self.tesselate(d_min, d_max)

        fig = plt.figure(figsize=(12, 12))

        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        ax2.grid(False)
        ax3 = fig.add_subplot(2, 2, 3)
        ax4 = fig.add_subplot(2, 2, 4)

        if equal_aspect:
            ax1.set_xlim([vert.min(), vert.max() + self.lbox])
            ax1.set_ylim([vert.min(), vert.max() + self.lbox])
            ax2.set_zlim3d(vert.min(), vert.max() + self.lbox)
            ax2.set_ylim3d(vert.min(), vert.max() + self.lbox)
            ax2.set_xlim3d(vert.min(), vert.max() + self.lbox)
            ax3.set_xlim([vert.min(), vert.max() + self.lbox])
            ax3.set_ylim([vert.min(), vert.max() + self.lbox])
            ax4.set_xlim([vert.min(), vert.max() + self.lbox])
            ax4.set_ylim([vert.min(), vert.max() + self.lbox])

        ax1.set_xlabel("X [cMpc]")
        ax1.set_ylabel("Y [cMpc]")
        ax2.set_xlabel("X [cMpc]")
        ax2.set_ylabel("Y [cMpc]")
        ax2.set_zlabel("Z [cMpc]")
        ax3.set_xlabel("X [cMpc]")
        ax3.set_ylabel("Z [cMpc]")
        ax4.set_xlabel("Y [cMpc]")
        ax4.set_ylabel("Z [cMpc]")

        if tesselations:
            for og in vert:
                render_cube(ax2, O=og, L=self.lbox)

                axis = [0, 1]
                rec = coordinate_plane(og, lbox=self.lbox, axes=axis)
                rec[2:, :] = rec[2:, :][::-1]
                rec = np.vstack((rec, rec[0, :]))
                ax1.plot(rec[:, axis[0]], rec[:, axis[1]], "b", alpha=0.25)

                axis = [0, 2]
                rec = coordinate_plane(og, lbox=self.lbox, axes=axis)
                rec[2:, :] = rec[2:, :][::-1]
                rec = np.vstack((rec, rec[0, :]))
                ax3.plot(rec[:, axis[0]], rec[:, axis[1]], "b", alpha=0.25)

                axis = [1, 2]
                rec = coordinate_plane(og, lbox=self.lbox, axes=axis)
                rec[2:, :] = rec[2:, :][::-1]
                rec = np.vstack((rec, rec[0, :]))
                ax4.plot(rec[:, axis[0]], rec[:, axis[1]], "b", alpha=0.25)

        if los:
            a = Arrow3D(
                [self.u3[0] * d_min, self.u3[0] * d_max],
                [self.u3[1] * d_min, self.u3[1] * d_max],
                [self.u3[2] * d_min, self.u3[2] * d_max],
                mutation_scale=20,
                lw=2,
                arrowstyle="-|>",
                color="r",
            )
            ax2.add_artist(a)

            x = self.u3[0] * d_min
            dx = np.diff([self.u3[0] * d_min, self.u3[0] * d_max])[0]
            y = self.u3[1] * d_min
            dy = np.diff([self.u3[1] * d_min, self.u3[1] * d_max])[0]
            ax1.arrow(x, y, dx, dy, lw=2, color="r")

            x = self.u3[0] * d_min
            dx = np.diff([self.u3[0] * d_min, self.u3[0] * d_max])[0]
            y = self.u3[2] * d_min
            dy = np.diff([self.u3[2] * d_min, self.u3[2] * d_max])[0]
            ax3.arrow(x, y, dx, dy, lw=2, color="r")

            x = self.u3[1] * d_min
            dx = np.diff([self.u3[1] * d_min, self.u3[1] * d_max])[0]
            y = self.u3[2] * d_min
            dy = np.diff([self.u3[2] * d_min, self.u3[2] * d_max])[0]
            ax4.arrow(x, y, dx, dy, lw=2, color="r")

        if cone_edges:
            r = [-0.5, 0.5]
            for i, r1 in enumerate(r):
                for j, r2 in enumerate(r):
                    vec = rotate(self.u3, angle=self.dd.value * r1, u=self.u1)
                    vec = rotate(vec, angle=self.da.value * r2, u=self.u2)
                    ax2.plot(
                        [vec[0] * d_min, vec[0] * d_max],
                        [vec[1] * d_min, vec[1] * d_max],
                        [vec[2] * d_min, vec[2] * d_max],
                        lw=2,
                        color="g",
                    )

                    x = vec[0] * d_min
                    dx = np.diff([vec[0] * d_min, vec[0] * d_max])[0]
                    y = vec[1] * d_min
                    dy = np.diff([vec[1] * d_min, vec[1] * d_max])[0]
                    ax1.arrow(x, y, dx, dy, lw=2, color="g")

                    x = vec[0] * d_min
                    dx = np.diff([vec[0] * d_min, vec[0] * d_max])[0]
                    y = vec[2] * d_min
                    dy = np.diff([vec[2] * d_min, vec[2] * d_max])[0]
                    ax3.arrow(x, y, dx, dy, lw=2, color="g")

                    x = vec[1] * d_min
                    dx = np.diff([vec[1] * d_min, vec[1] * d_max])[0]
                    y = vec[2] * d_min
                    dy = np.diff([vec[2] * d_min, vec[2] * d_max])[0]
                    ax4.arrow(x, y, dx, dy, lw=2, color="g")
