use glam::{U8Vec3, UVec3};
use thiserror::Error;

use crate::{
    math::Aabb,
    voxel::{Voxel, VoxelId},
};

pub type VoxelBlockData = Box<[Voxel; (VoxelBlock::WIDTH as usize).pow(3)]>;

#[derive(Debug, Clone, PartialEq)]
pub struct VoxelBlock {
    data: VoxelBlockData,
    bounds: Aabb,
}

impl VoxelBlock {
    pub const WIDTH: u8 = 16;
    pub const AREA: u16 = (Self::WIDTH as u16).pow(2);
    pub const VOLUME: u32 = Self::AREA as u32 * Self::WIDTH as u32;

    pub fn new(data: VoxelBlockData, coords: UVec3) -> Self {
        let coords = coords.as_vec3();
        Self {
            data,
            bounds: Aabb::new(coords, coords + Self::WIDTH as f32),
        }
    }

    pub fn get(&self, pos: U8Vec3) -> &Voxel {
        let index = Self::to_index(pos);
        &self.data[index]
    }

    pub fn get_mut(&mut self, pos: U8Vec3) -> &mut Voxel {
        let index = Self::to_index(pos);
        &mut self.data[index]
    }

    fn to_index(pos: U8Vec3) -> usize {
        debug_assert!(
            pos.x < Self::WIDTH && pos.y < Self::WIDTH && pos.z < Self::WIDTH,
            "coordinates out of bounds"
        );
        let width = Self::WIDTH as usize;
        let area = Self::AREA as usize;
        pos.x as usize + pos.z as usize * width + pos.y as usize * area
    }

    pub fn to_rle(&self) -> Vec<Rle> {
        let mut rle = Vec::new();

        let mut prev_voxel = self.data[0];
        let mut count = 1;

        for &voxel in self.data.iter().skip(1) {
            if prev_voxel == voxel {
                count += 1;
            } else {
                rle.push((count, voxel as VoxelId));
                count = 0;
            }
            prev_voxel = voxel;
        }
        rle.push((count, prev_voxel as VoxelId));
        rle
    }

    pub fn from_rle<I>(rle: I, coords: UVec3) -> Result<Self, RleError>
    where
        I: IntoIterator<Item = Rle>,
    {
        let mut voxels = Vec::with_capacity(Self::VOLUME as usize);

        for (count, id) in rle.into_iter() {
            let voxel = Voxel::ALL
                .get(id as usize)
                .ok_or(RleError::InvalidVoxelId(id))?;
            voxels.extend(vec![*voxel; count as usize]);
        }

        let data = voxels.try_into().map_err(|_| RleError::InvalidShape)?;
        Ok(Self::new(data, coords))
    }
}

pub type Rle = (VoxelCount, VoxelId);

pub type VoxelCount = u32;

#[derive(Error, Debug)]
pub enum RleError {
    #[error("invalid voxel ID: {0}")]
    InvalidVoxelId(VoxelId),
    #[error(
        "shape of RLE could not be converted to an array of size {}",
        VoxelBlock::VOLUME
    )]
    InvalidShape,
}
