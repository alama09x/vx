use std::collections::BTreeMap;

use ash::vk;
use data::{IntoBytes, IntoBytesMut};

pub struct Mesh {
    primitive_topology: vk::PrimitiveTopology,
    attributes: BTreeMap<MeshVertexAttributeId, MeshAttributeData>,
    indices: Option<Indices>,
}

impl Mesh {
    pub const ATTRIBUTE_POSITION: MeshVertexAttribute =
        MeshVertexAttribute::new("Vertex_Position", 0, VertexFormat::Float32x3);
    pub const ATTRIBUTE_NORMAL: MeshVertexAttribute =
        MeshVertexAttribute::new("Vertex_Normal", 1, VertexFormat::Float32x3);
    pub const ATTRIBUTE_COLOR: MeshVertexAttribute =
        MeshVertexAttribute::new("Vertex_Color", 2, VertexFormat::Float32x3);
    pub const ATTRIBUTE_UV: MeshVertexAttribute =
        MeshVertexAttribute::new("Vertex_UV", 2, VertexFormat::Float32x2);

    pub fn new(primitive_topology: vk::PrimitiveTopology) -> Self {
        Self {
            primitive_topology,
            attributes: BTreeMap::new(),
            indices: None,
        }
    }

    pub fn primitive_topology(&self) -> vk::PrimitiveTopology {
        self.primitive_topology
    }

    pub fn insert_attribute(
        &mut self,
        attribute: MeshVertexAttribute,
        data: MeshAttributeData,
    ) -> Option<MeshAttributeData> {
        if attribute.format != (&data).into() {
            panic!("Error: invalid data for attribute: {}", attribute.name);
        }
        self.attributes.insert(attribute.id, data)
    }

    pub fn with_inserted_attribute(
        mut self,
        attribute: MeshVertexAttribute,
        data: MeshAttributeData,
    ) -> Self {
        self.insert_attribute(attribute, data);
        self
    }

    pub fn set_indices(&mut self, indices: Option<Indices>) {
        self.indices = indices;
    }

    pub fn with_indices(mut self, indices: Option<Indices>) -> Self {
        self.set_indices(indices);
        self
    }

    pub fn binding_description(&self) -> vk::VertexInputBindingDescription {
        todo!()
    }

    pub fn attribute_descriptions(&self) -> Vec<vk::VertexInputAttributeDescription> {
        let mut accumulative_offset = 0;
        self.attributes
            .values()
            .enumerate()
            .map(|(i, data)| {
                let result = vk::VertexInputAttributeDescription::default()
                    .binding(i as u32)
                    .offset(accumulative_offset)
                    .format(data.into());
                accumulative_offset += data.size() as u32;
                result
            })
            .collect()
    }
}

pub struct VertexAttribute {
    pub format: VertexFormat,
    pub offset: u64,
    pub shader_location: u32,
}

pub struct MeshVertexAttribute {
    pub name: &'static str,
    pub id: MeshVertexAttributeId,
    pub format: VertexFormat,
}

impl MeshVertexAttribute {
    pub const fn new(name: &'static str, id: u32, format: VertexFormat) -> Self {
        Self {
            name,
            id: MeshVertexAttributeId::new(id),
            format,
        }
    }
}

pub enum VertexStepMode {
    Vertex,
    Instance,
}

pub struct VertexBufferLayout {
    pub array_stride: u64,
    pub step_mode: VertexStepMode,
    pub attributes: Vec<VertexAttribute>,
}

impl VertexBufferLayout {
    pub fn from_formats<T: IntoIterator<Item = VertexFormat>>(
        step_mode: VertexStepMode,
        vertex_formats: T,
    ) -> Self {
        let mut offset = 0;
        let attributes = vertex_formats
            .into_iter()
            .enumerate()
            .map(|(loc, format)| {
                let attrib = VertexAttribute {
                    format,
                    offset,
                    shader_location: loc as u32,
                };
                offset += format.size() as u64;
                attrib
            })
            .collect();

        Self {
            array_stride: offset,
            step_mode,
            attributes,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshVertexAttributeId(u32);

impl MeshVertexAttributeId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
}
#[test]
fn construct_mesh() {}

#[derive(Debug, Clone)]
pub enum Indices {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

#[derive(Debug, Clone)]
pub enum MeshAttributeData {
    Float32(Vec<f32>),
    Sint32(Vec<i32>),
    Uint32(Vec<u32>),
    Float32x2(Vec<[f32; 2]>),
    Sint32x2(Vec<[i32; 2]>),
    Uint32x2(Vec<[u32; 2]>),
    Float32x3(Vec<[f32; 3]>),
    Sint32x3(Vec<[i32; 3]>),
    Uint32x3(Vec<[u32; 3]>),
    Float32x4(Vec<[f32; 4]>),
    Sint32x4(Vec<[i32; 4]>),
    Uint32x4(Vec<[u32; 4]>),
    Sint16x2(Vec<[i16; 2]>),
    Snorm16x2(Vec<[i16; 2]>),
    Uint16x2(Vec<[u16; 2]>),
    Unorm16x2(Vec<[u16; 2]>),
    Sint16x4(Vec<[i16; 4]>),
    Snorm16x4(Vec<[i16; 4]>),
    Uint16x4(Vec<[u16; 4]>),
    Unorm16x4(Vec<[u16; 4]>),
    Sint8x2(Vec<[i8; 2]>),
    Snorm8x2(Vec<[i8; 2]>),
    Uint8x2(Vec<[u8; 2]>),
    Unorm8x2(Vec<[u8; 2]>),
    Sint8x4(Vec<[i8; 4]>),
    Snorm8x4(Vec<[i8; 4]>),
    Uint8x4(Vec<[u8; 4]>),
    Unorm8x4(Vec<[u8; 4]>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexFormat {
    Float32,
    Sint32,
    Uint32,
    Float32x2,
    Sint32x2,
    Uint32x2,
    Float32x3,
    Sint32x3,
    Uint32x3,
    Float32x4,
    Sint32x4,
    Uint32x4,
    Sint16x2,
    Snorm16x2,
    Uint16x2,
    Unorm16x2,
    Sint16x4,
    Snorm16x4,
    Uint16x4,
    Unorm16x4,
    Sint8x2,
    Snorm8x2,
    Uint8x2,
    Unorm8x2,
    Sint8x4,
    Snorm8x4,
    Uint8x4,
    Unorm8x4,
}

impl VertexFormat {
    pub fn size(&self) -> usize {
        match self {
            Self::Float32 => 4,
            Self::Sint32 => 4,
            Self::Uint32 => 4,
            Self::Float32x2 => 4 * 2,
            Self::Sint32x2 => 4 * 2,
            Self::Uint32x2 => 4 * 2,
            Self::Float32x3 => 4 * 3,
            Self::Sint32x3 => 4 * 3,
            Self::Uint32x3 => 4 * 3,
            Self::Float32x4 => 4 * 4,
            Self::Sint32x4 => 4 * 4,
            Self::Uint32x4 => 4 * 4,
            Self::Sint16x2 => 2 * 2,
            Self::Snorm16x2 => 2 * 2,
            Self::Uint16x2 => 2 * 2,
            Self::Unorm16x2 => 2 * 2,
            Self::Sint16x4 => 2 * 4,
            Self::Snorm16x4 => 2 * 4,
            Self::Uint16x4 => 2 * 4,
            Self::Unorm16x4 => 2 * 4,
            Self::Sint8x2 => 2,
            Self::Snorm8x2 => 2,
            Self::Uint8x2 => 2,
            Self::Unorm8x2 => 2,
            Self::Sint8x4 => 4,
            Self::Snorm8x4 => 4,
            Self::Uint8x4 => 4,
            Self::Unorm8x4 => 4,
        }
    }
}

impl MeshAttributeData {
    pub fn size(&self) -> usize {
        match self {
            Self::Float32(_) => 4,
            Self::Sint32(_) => 4,
            Self::Uint32(_) => 4,
            Self::Float32x2(_) => 4 * 2,
            Self::Sint32x2(_) => 4 * 2,
            Self::Uint32x2(_) => 4 * 2,
            Self::Float32x3(_) => 4 * 3,
            Self::Sint32x3(_) => 4 * 3,
            Self::Uint32x3(_) => 4 * 3,
            Self::Float32x4(_) => 4 * 4,
            Self::Sint32x4(_) => 4 * 4,
            Self::Uint32x4(_) => 4 * 4,
            Self::Sint16x2(_) => 2 * 2,
            Self::Snorm16x2(_) => 2 * 2,
            Self::Uint16x2(_) => 2 * 2,
            Self::Unorm16x2(_) => 2 * 2,
            Self::Sint16x4(_) => 2 * 4,
            Self::Snorm16x4(_) => 2 * 4,
            Self::Uint16x4(_) => 2 * 4,
            Self::Unorm16x4(_) => 2 * 4,
            Self::Sint8x2(_) => 2,
            Self::Snorm8x2(_) => 2,
            Self::Uint8x2(_) => 2,
            Self::Unorm8x2(_) => 2,
            Self::Sint8x4(_) => 4,
            Self::Snorm8x4(_) => 4,
            Self::Uint8x4(_) => 4,
            Self::Unorm8x4(_) => 4,
        }
    }
}

impl From<MeshAttributeData> for VertexFormat {
    fn from(value: MeshAttributeData) -> Self {
        (&value).into()
    }
}

impl From<&MeshAttributeData> for VertexFormat {
    fn from(value: &MeshAttributeData) -> Self {
        match value {
            MeshAttributeData::Float32(_) => VertexFormat::Float32,
            MeshAttributeData::Sint32(_) => VertexFormat::Sint32,
            MeshAttributeData::Uint32(_) => VertexFormat::Uint32,
            MeshAttributeData::Float32x2(_) => VertexFormat::Float32x2,
            MeshAttributeData::Sint32x2(_) => VertexFormat::Sint32x2,
            MeshAttributeData::Uint32x2(_) => VertexFormat::Uint32x2,
            MeshAttributeData::Float32x3(_) => VertexFormat::Float32x3,
            MeshAttributeData::Sint32x3(_) => VertexFormat::Sint32x3,
            MeshAttributeData::Uint32x3(_) => VertexFormat::Uint32x3,
            MeshAttributeData::Float32x4(_) => VertexFormat::Float32x4,
            MeshAttributeData::Sint32x4(_) => VertexFormat::Sint32x4,
            MeshAttributeData::Uint32x4(_) => VertexFormat::Uint32x4,
            MeshAttributeData::Sint16x2(_) => VertexFormat::Sint16x2,
            MeshAttributeData::Snorm16x2(_) => VertexFormat::Snorm16x2,
            MeshAttributeData::Uint16x2(_) => VertexFormat::Uint16x2,
            MeshAttributeData::Unorm16x2(_) => VertexFormat::Unorm16x2,
            MeshAttributeData::Sint16x4(_) => VertexFormat::Sint16x4,
            MeshAttributeData::Snorm16x4(_) => VertexFormat::Snorm16x4,
            MeshAttributeData::Uint16x4(_) => VertexFormat::Uint16x4,
            MeshAttributeData::Unorm16x4(_) => VertexFormat::Unorm16x4,
            MeshAttributeData::Sint8x2(_) => VertexFormat::Sint8x2,
            MeshAttributeData::Snorm8x2(_) => VertexFormat::Snorm8x2,
            MeshAttributeData::Uint8x2(_) => VertexFormat::Uint8x2,
            MeshAttributeData::Unorm8x2(_) => VertexFormat::Unorm8x2,
            MeshAttributeData::Sint8x4(_) => VertexFormat::Sint8x4,
            MeshAttributeData::Snorm8x4(_) => VertexFormat::Snorm8x4,
            MeshAttributeData::Uint8x4(_) => VertexFormat::Uint8x4,
            MeshAttributeData::Unorm8x4(_) => VertexFormat::Unorm8x4,
        }
    }
}

impl From<VertexFormat> for vk::Format {
    fn from(format: VertexFormat) -> Self {
        (&format).into()
    }
}

impl From<&VertexFormat> for vk::Format {
    fn from(format: &VertexFormat) -> Self {
        match format {
            VertexFormat::Float32 => vk::Format::R32_SFLOAT,
            VertexFormat::Sint32 => vk::Format::R32_SINT,
            VertexFormat::Uint32 => vk::Format::R32_UINT,
            VertexFormat::Float32x2 => vk::Format::R32G32_SFLOAT,
            VertexFormat::Sint32x2 => vk::Format::R32G32_SINT,
            VertexFormat::Uint32x2 => vk::Format::R32G32_UINT,
            VertexFormat::Float32x3 => vk::Format::R32G32B32_SFLOAT,
            VertexFormat::Sint32x3 => vk::Format::R32G32B32_SINT,
            VertexFormat::Uint32x3 => vk::Format::R32G32B32_UINT,
            VertexFormat::Float32x4 => vk::Format::R32G32B32A32_SFLOAT,
            VertexFormat::Sint32x4 => vk::Format::R32G32B32A32_SINT,
            VertexFormat::Uint32x4 => vk::Format::R32G32B32A32_SINT,
            VertexFormat::Sint16x2 => vk::Format::R16G16_SINT,
            VertexFormat::Snorm16x2 => vk::Format::R16G16_SNORM,
            VertexFormat::Uint16x2 => vk::Format::R16G16_UINT,
            VertexFormat::Unorm16x2 => vk::Format::R16G16_UNORM,
            VertexFormat::Sint16x4 => vk::Format::R16G16B16A16_SINT,
            VertexFormat::Snorm16x4 => vk::Format::R16G16B16A16_SNORM,
            VertexFormat::Uint16x4 => vk::Format::R16G16B16A16_UINT,
            VertexFormat::Unorm16x4 => vk::Format::R16G16B16A16_UNORM,
            VertexFormat::Sint8x2 => vk::Format::R8G8_SINT,
            VertexFormat::Snorm8x2 => vk::Format::R8G8_SNORM,
            VertexFormat::Uint8x2 => vk::Format::R8G8_UINT,
            VertexFormat::Unorm8x2 => vk::Format::R8G8_UNORM,
            VertexFormat::Sint8x4 => vk::Format::R8G8B8A8_SINT,
            VertexFormat::Snorm8x4 => vk::Format::R8G8B8A8_SNORM,
            VertexFormat::Uint8x4 => vk::Format::R8G8B8A8_UINT,
            VertexFormat::Unorm8x4 => vk::Format::R8G8B8A8_UNORM,
        }
    }
}

impl From<MeshAttributeData> for vk::Format {
    fn from(data: MeshAttributeData) -> Self {
        (&data).into()
    }
}

impl From<&MeshAttributeData> for vk::Format {
    fn from(data: &MeshAttributeData) -> Self {
        match data {
            MeshAttributeData::Float32(_) => vk::Format::R32_SFLOAT,
            MeshAttributeData::Sint32(_) => vk::Format::R32_SINT,
            MeshAttributeData::Uint32(_) => vk::Format::R32_UINT,
            MeshAttributeData::Float32x2(_) => vk::Format::R32G32_SFLOAT,
            MeshAttributeData::Sint32x2(_) => vk::Format::R32G32_SINT,
            MeshAttributeData::Uint32x2(_) => vk::Format::R32G32_UINT,
            MeshAttributeData::Float32x3(_) => vk::Format::R32G32B32_SFLOAT,
            MeshAttributeData::Sint32x3(_) => vk::Format::R32G32B32_SINT,
            MeshAttributeData::Uint32x3(_) => vk::Format::R32G32B32_UINT,
            MeshAttributeData::Float32x4(_) => vk::Format::R32G32B32A32_SFLOAT,
            MeshAttributeData::Sint32x4(_) => vk::Format::R32G32B32A32_SINT,
            MeshAttributeData::Uint32x4(_) => vk::Format::R32G32B32A32_SINT,
            MeshAttributeData::Sint16x2(_) => vk::Format::R16G16_SINT,
            MeshAttributeData::Snorm16x2(_) => vk::Format::R16G16_SNORM,
            MeshAttributeData::Uint16x2(_) => vk::Format::R16G16_UINT,
            MeshAttributeData::Unorm16x2(_) => vk::Format::R16G16_UNORM,
            MeshAttributeData::Sint16x4(_) => vk::Format::R16G16B16A16_SINT,
            MeshAttributeData::Snorm16x4(_) => vk::Format::R16G16B16A16_SNORM,
            MeshAttributeData::Uint16x4(_) => vk::Format::R16G16B16A16_UINT,
            MeshAttributeData::Unorm16x4(_) => vk::Format::R16G16B16A16_UNORM,
            MeshAttributeData::Sint8x2(_) => vk::Format::R8G8_SINT,
            MeshAttributeData::Snorm8x2(_) => vk::Format::R8G8_SNORM,
            MeshAttributeData::Uint8x2(_) => vk::Format::R8G8_UINT,
            MeshAttributeData::Unorm8x2(_) => vk::Format::R8G8_UNORM,
            MeshAttributeData::Sint8x4(_) => vk::Format::R8G8B8A8_SINT,
            MeshAttributeData::Snorm8x4(_) => vk::Format::R8G8B8A8_SNORM,
            MeshAttributeData::Uint8x4(_) => vk::Format::R8G8B8A8_UINT,
            MeshAttributeData::Unorm8x4(_) => vk::Format::R8G8B8A8_UNORM,
        }
    }
}

impl IntoBytes for MeshAttributeData {
    fn to_bytes(&self) -> &[u8] {
        match self {
            Self::Float32(data) => bytemuck::cast_slice(data),
            Self::Sint32(data) => bytemuck::cast_slice(data),
            Self::Uint32(data) => bytemuck::cast_slice(data),
            Self::Float32x2(data) => bytemuck::cast_slice(data),
            Self::Sint32x2(data) => bytemuck::cast_slice(data),
            Self::Uint32x2(data) => bytemuck::cast_slice(data),
            Self::Float32x3(data) => bytemuck::cast_slice(data),
            Self::Sint32x3(data) => bytemuck::cast_slice(data),
            Self::Uint32x3(data) => bytemuck::cast_slice(data),
            Self::Float32x4(data) => bytemuck::cast_slice(data),
            Self::Sint32x4(data) => bytemuck::cast_slice(data),
            Self::Uint32x4(data) => bytemuck::cast_slice(data),
            Self::Sint16x2(data) => bytemuck::cast_slice(data),
            Self::Snorm16x2(data) => bytemuck::cast_slice(data),
            Self::Uint16x2(data) => bytemuck::cast_slice(data),
            Self::Unorm16x2(data) => bytemuck::cast_slice(data),
            Self::Sint16x4(data) => bytemuck::cast_slice(data),
            Self::Snorm16x4(data) => bytemuck::cast_slice(data),
            Self::Uint16x4(data) => bytemuck::cast_slice(data),
            Self::Unorm16x4(data) => bytemuck::cast_slice(data),
            Self::Sint8x2(data) => bytemuck::cast_slice(data),
            Self::Snorm8x2(data) => bytemuck::cast_slice(data),
            Self::Uint8x2(data) => bytemuck::cast_slice(data),
            Self::Unorm8x2(data) => bytemuck::cast_slice(data),
            Self::Sint8x4(data) => bytemuck::cast_slice(data),
            Self::Snorm8x4(data) => bytemuck::cast_slice(data),
            Self::Uint8x4(data) => bytemuck::cast_slice(data),
            Self::Unorm8x4(data) => bytemuck::cast_slice(data),
        }
    }
}

impl IntoBytesMut for MeshAttributeData {
    fn to_bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Self::Float32(data) => bytemuck::cast_slice_mut(data),
            Self::Sint32(data) => bytemuck::cast_slice_mut(data),
            Self::Uint32(data) => bytemuck::cast_slice_mut(data),
            Self::Float32x2(data) => bytemuck::cast_slice_mut(data),
            Self::Sint32x2(data) => bytemuck::cast_slice_mut(data),
            Self::Uint32x2(data) => bytemuck::cast_slice_mut(data),
            Self::Float32x3(data) => bytemuck::cast_slice_mut(data),
            Self::Sint32x3(data) => bytemuck::cast_slice_mut(data),
            Self::Uint32x3(data) => bytemuck::cast_slice_mut(data),
            Self::Float32x4(data) => bytemuck::cast_slice_mut(data),
            Self::Sint32x4(data) => bytemuck::cast_slice_mut(data),
            Self::Uint32x4(data) => bytemuck::cast_slice_mut(data),
            Self::Sint16x2(data) => bytemuck::cast_slice_mut(data),
            Self::Snorm16x2(data) => bytemuck::cast_slice_mut(data),
            Self::Uint16x2(data) => bytemuck::cast_slice_mut(data),
            Self::Unorm16x2(data) => bytemuck::cast_slice_mut(data),
            Self::Sint16x4(data) => bytemuck::cast_slice_mut(data),
            Self::Snorm16x4(data) => bytemuck::cast_slice_mut(data),
            Self::Uint16x4(data) => bytemuck::cast_slice_mut(data),
            Self::Unorm16x4(data) => bytemuck::cast_slice_mut(data),
            Self::Sint8x2(data) => bytemuck::cast_slice_mut(data),
            Self::Snorm8x2(data) => bytemuck::cast_slice_mut(data),
            Self::Uint8x2(data) => bytemuck::cast_slice_mut(data),
            Self::Unorm8x2(data) => bytemuck::cast_slice_mut(data),
            Self::Sint8x4(data) => bytemuck::cast_slice_mut(data),
            Self::Snorm8x4(data) => bytemuck::cast_slice_mut(data),
            Self::Uint8x4(data) => bytemuck::cast_slice_mut(data),
            Self::Unorm8x4(data) => bytemuck::cast_slice_mut(data),
        }
    }
}
