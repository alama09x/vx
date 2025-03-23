use std::collections::BTreeMap;

use ash::vk;
use data::{IntoBytes, IntoBytesMut};

#[derive(Debug, Clone)]
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
        values: impl Into<VertexAttributeValues>,
    ) -> Option<MeshAttributeData> {
        let values = values.into();
        let values_format = VertexFormat::from(&values);
        if values_format != attribute.format {
            panic!("Error: invalid data for attribute: {}", attribute.name);
        }
        self.attributes
            .insert(attribute.id, MeshAttributeData { attribute, values })
    }

    pub fn with_inserted_attribute(
        mut self,
        attribute: MeshVertexAttribute,
        data: impl Into<VertexAttributeValues>,
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

    pub fn binding_description(
        &self,
        binding: u32,
        input_rate: vk::VertexInputRate,
    ) -> vk::VertexInputBindingDescription {
        let offset: u64 = self
            .attributes
            .values()
            .map(|data| data.attribute.format.size())
            .sum();

        vk::VertexInputBindingDescription::default()
            .binding(binding)
            .input_rate(input_rate)
            .stride(offset as u32)
    }

    pub fn attribute_descriptions(&self, binding: u32) -> Vec<vk::VertexInputAttributeDescription> {
        let mut accumulative_offset = 0;
        self.attributes
            .values()
            .enumerate()
            .map(|(i, data)| {
                let result = vk::VertexInputAttributeDescription::default()
                    .binding(binding)
                    .location(i as u32)
                    .offset(accumulative_offset)
                    .format(data.attribute.format.into());
                accumulative_offset += data.attribute.format.size() as u32;
                result
            })
            .collect()
    }

    pub fn vertex_count(&self) -> usize {
        let mut count: Option<usize> = None;
        for (attribute_id, attribute_data) in &self.attributes {
            let attribute_len = attribute_data.values.len();
            if let Some(previous_count) = count {
                if previous_count != attribute_len {
                    let name = self
                        .attributes
                        .get(attribute_id)
                        .map(|data| data.attribute.name.to_string())
                        .unwrap_or_else(|| format!("{attribute_id:?}"));
                    eprintln!("Attribute {name} has a different vertex count ({attribute_len}) than others in the mesh; truncating to smallest");
                    count = Some(previous_count.min(attribute_len))
                }
            } else {
                count = Some(attribute_len);
            }
        }
        count.unwrap_or(0)
    }

    pub fn vertex_size(&self) -> u64 {
        self.attributes
            .values()
            .map(|data| data.attribute.format.size())
            .sum()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MeshVertexAttributeId(u32);

impl MeshVertexAttributeId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
}

#[derive(Debug, Clone)]
pub struct MeshAttributeData {
    pub attribute: MeshVertexAttribute,
    pub values: VertexAttributeValues,
}

#[derive(Debug, Clone, Copy)]
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

pub struct VertexBufferLayout {
    pub stride: u64,
    pub input_rate: vk::VertexInputRate,
    pub attributes: Vec<VertexAttribute>,
}

impl VertexBufferLayout {
    pub fn from_formats<T: IntoIterator<Item = VertexFormat>>(
        input_rate: vk::VertexInputRate,
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
                    location: loc as u32,
                };
                offset += format.size();
                attrib
            })
            .collect();

        Self {
            stride: offset,
            input_rate,
            attributes,
        }
    }
}

pub struct VertexAttribute {
    pub format: VertexFormat,
    pub offset: u64,
    pub location: u32,
}
#[test]
fn construct_mesh() {
    let positions = vec![[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0; 3]];
    let normals = vec![[1.0, 0.0, 0.0]; 3];
    let mesh = Mesh::new(vk::PrimitiveTopology::TRIANGLE_LIST)
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
        .with_indices(Some(Indices::U16(vec![0, 1, 2, 0, 2, 3])));

    println!(
        "Attribute Descriptions {:#?}",
        mesh.attribute_descriptions(0)
    );
    println!(
        "Binding Description: {:#?}",
        mesh.binding_description(0, vk::VertexInputRate::VERTEX)
    );
    println!("Vertex Buffer Size: {}", mesh.vertex_size());
}

#[derive(Debug, Clone)]
pub enum Indices {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

#[derive(Debug, Clone)]
pub enum VertexAttributeValues {
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

macro_rules! impl_from {
    ($from:ty, $variant:tt) => {
        impl From<Vec<$from>> for VertexAttributeValues {
            fn from(vec: Vec<$from>) -> Self {
                VertexAttributeValues::$variant(vec)
            }
        }
    };
}

// TODO: Finish implementing these
impl_from!(f32, Float32);
impl_from!(i32, Sint32);
impl_from!(u32, Uint32);
impl_from!([f32; 2], Float32x2);
impl_from!([f32; 3], Float32x3);

impl VertexAttributeValues {
    pub fn len(&self) -> usize {
        match self {
            Self::Float32(data) => data.len(),
            Self::Sint32(data) => data.len(),
            Self::Uint32(data) => data.len(),
            Self::Float32x2(data) => data.len(),
            Self::Sint32x2(data) => data.len(),
            Self::Uint32x2(data) => data.len(),
            Self::Float32x3(data) => data.len(),
            Self::Sint32x3(data) => data.len(),
            Self::Uint32x3(data) => data.len(),
            Self::Float32x4(data) => data.len(),
            Self::Sint32x4(data) => data.len(),
            Self::Uint32x4(data) => data.len(),
            Self::Sint16x2(data) => data.len(),
            Self::Snorm16x2(data) => data.len(),
            Self::Uint16x2(data) => data.len(),
            Self::Unorm16x2(data) => data.len(),
            Self::Sint16x4(data) => data.len(),
            Self::Snorm16x4(data) => data.len(),
            Self::Uint16x4(data) => data.len(),
            Self::Unorm16x4(data) => data.len(),
            Self::Sint8x2(data) => data.len(),
            Self::Snorm8x2(data) => data.len(),
            Self::Uint8x2(data) => data.len(),
            Self::Unorm8x2(data) => data.len(),
            Self::Sint8x4(data) => data.len(),
            Self::Snorm8x4(data) => data.len(),
            Self::Uint8x4(data) => data.len(),
            Self::Unorm8x4(data) => data.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
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
    pub fn size(&self) -> u64 {
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

impl From<VertexAttributeValues> for VertexFormat {
    fn from(value: VertexAttributeValues) -> Self {
        (&value).into()
    }
}

impl From<&VertexAttributeValues> for VertexFormat {
    fn from(values: &VertexAttributeValues) -> Self {
        match values {
            VertexAttributeValues::Float32(_) => VertexFormat::Float32,
            VertexAttributeValues::Sint32(_) => VertexFormat::Sint32,
            VertexAttributeValues::Uint32(_) => VertexFormat::Uint32,
            VertexAttributeValues::Float32x2(_) => VertexFormat::Float32x2,
            VertexAttributeValues::Sint32x2(_) => VertexFormat::Sint32x2,
            VertexAttributeValues::Uint32x2(_) => VertexFormat::Uint32x2,
            VertexAttributeValues::Float32x3(_) => VertexFormat::Float32x3,
            VertexAttributeValues::Sint32x3(_) => VertexFormat::Sint32x3,
            VertexAttributeValues::Uint32x3(_) => VertexFormat::Uint32x3,
            VertexAttributeValues::Float32x4(_) => VertexFormat::Float32x4,
            VertexAttributeValues::Sint32x4(_) => VertexFormat::Sint32x4,
            VertexAttributeValues::Uint32x4(_) => VertexFormat::Uint32x4,
            VertexAttributeValues::Sint16x2(_) => VertexFormat::Sint16x2,
            VertexAttributeValues::Snorm16x2(_) => VertexFormat::Snorm16x2,
            VertexAttributeValues::Uint16x2(_) => VertexFormat::Uint16x2,
            VertexAttributeValues::Unorm16x2(_) => VertexFormat::Unorm16x2,
            VertexAttributeValues::Sint16x4(_) => VertexFormat::Sint16x4,
            VertexAttributeValues::Snorm16x4(_) => VertexFormat::Snorm16x4,
            VertexAttributeValues::Uint16x4(_) => VertexFormat::Uint16x4,
            VertexAttributeValues::Unorm16x4(_) => VertexFormat::Unorm16x4,
            VertexAttributeValues::Sint8x2(_) => VertexFormat::Sint8x2,
            VertexAttributeValues::Snorm8x2(_) => VertexFormat::Snorm8x2,
            VertexAttributeValues::Uint8x2(_) => VertexFormat::Uint8x2,
            VertexAttributeValues::Unorm8x2(_) => VertexFormat::Unorm8x2,
            VertexAttributeValues::Sint8x4(_) => VertexFormat::Sint8x4,
            VertexAttributeValues::Snorm8x4(_) => VertexFormat::Snorm8x4,
            VertexAttributeValues::Uint8x4(_) => VertexFormat::Uint8x4,
            VertexAttributeValues::Unorm8x4(_) => VertexFormat::Unorm8x4,
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

impl From<VertexAttributeValues> for vk::Format {
    fn from(data: VertexAttributeValues) -> Self {
        (&data).into()
    }
}

impl From<&VertexAttributeValues> for vk::Format {
    fn from(values: &VertexAttributeValues) -> Self {
        match values {
            VertexAttributeValues::Float32(_) => vk::Format::R32_SFLOAT,
            VertexAttributeValues::Sint32(_) => vk::Format::R32_SINT,
            VertexAttributeValues::Uint32(_) => vk::Format::R32_UINT,
            VertexAttributeValues::Float32x2(_) => vk::Format::R32G32_SFLOAT,
            VertexAttributeValues::Sint32x2(_) => vk::Format::R32G32_SINT,
            VertexAttributeValues::Uint32x2(_) => vk::Format::R32G32_UINT,
            VertexAttributeValues::Float32x3(_) => vk::Format::R32G32B32_SFLOAT,
            VertexAttributeValues::Sint32x3(_) => vk::Format::R32G32B32_SINT,
            VertexAttributeValues::Uint32x3(_) => vk::Format::R32G32B32_UINT,
            VertexAttributeValues::Float32x4(_) => vk::Format::R32G32B32A32_SFLOAT,
            VertexAttributeValues::Sint32x4(_) => vk::Format::R32G32B32A32_SINT,
            VertexAttributeValues::Uint32x4(_) => vk::Format::R32G32B32A32_SINT,
            VertexAttributeValues::Sint16x2(_) => vk::Format::R16G16_SINT,
            VertexAttributeValues::Snorm16x2(_) => vk::Format::R16G16_SNORM,
            VertexAttributeValues::Uint16x2(_) => vk::Format::R16G16_UINT,
            VertexAttributeValues::Unorm16x2(_) => vk::Format::R16G16_UNORM,
            VertexAttributeValues::Sint16x4(_) => vk::Format::R16G16B16A16_SINT,
            VertexAttributeValues::Snorm16x4(_) => vk::Format::R16G16B16A16_SNORM,
            VertexAttributeValues::Uint16x4(_) => vk::Format::R16G16B16A16_UINT,
            VertexAttributeValues::Unorm16x4(_) => vk::Format::R16G16B16A16_UNORM,
            VertexAttributeValues::Sint8x2(_) => vk::Format::R8G8_SINT,
            VertexAttributeValues::Snorm8x2(_) => vk::Format::R8G8_SNORM,
            VertexAttributeValues::Uint8x2(_) => vk::Format::R8G8_UINT,
            VertexAttributeValues::Unorm8x2(_) => vk::Format::R8G8_UNORM,
            VertexAttributeValues::Sint8x4(_) => vk::Format::R8G8B8A8_SINT,
            VertexAttributeValues::Snorm8x4(_) => vk::Format::R8G8B8A8_SNORM,
            VertexAttributeValues::Uint8x4(_) => vk::Format::R8G8B8A8_UINT,
            VertexAttributeValues::Unorm8x4(_) => vk::Format::R8G8B8A8_UNORM,
        }
    }
}

impl IntoBytes for VertexAttributeValues {
    fn to_bytes(&self) -> &[u8] {
        match self {
            Self::Float32(values) => bytemuck::cast_slice(values),
            Self::Sint32(values) => bytemuck::cast_slice(values),
            Self::Uint32(values) => bytemuck::cast_slice(values),
            Self::Float32x2(values) => bytemuck::cast_slice(values),
            Self::Sint32x2(values) => bytemuck::cast_slice(values),
            Self::Uint32x2(values) => bytemuck::cast_slice(values),
            Self::Float32x3(values) => bytemuck::cast_slice(values),
            Self::Sint32x3(values) => bytemuck::cast_slice(values),
            Self::Uint32x3(values) => bytemuck::cast_slice(values),
            Self::Float32x4(values) => bytemuck::cast_slice(values),
            Self::Sint32x4(values) => bytemuck::cast_slice(values),
            Self::Uint32x4(values) => bytemuck::cast_slice(values),
            Self::Sint16x2(values) => bytemuck::cast_slice(values),
            Self::Snorm16x2(values) => bytemuck::cast_slice(values),
            Self::Uint16x2(values) => bytemuck::cast_slice(values),
            Self::Unorm16x2(values) => bytemuck::cast_slice(values),
            Self::Sint16x4(values) => bytemuck::cast_slice(values),
            Self::Snorm16x4(values) => bytemuck::cast_slice(values),
            Self::Uint16x4(values) => bytemuck::cast_slice(values),
            Self::Unorm16x4(values) => bytemuck::cast_slice(values),
            Self::Sint8x2(values) => bytemuck::cast_slice(values),
            Self::Snorm8x2(values) => bytemuck::cast_slice(values),
            Self::Uint8x2(values) => bytemuck::cast_slice(values),
            Self::Unorm8x2(values) => bytemuck::cast_slice(values),
            Self::Sint8x4(values) => bytemuck::cast_slice(values),
            Self::Snorm8x4(values) => bytemuck::cast_slice(values),
            Self::Uint8x4(values) => bytemuck::cast_slice(values),
            Self::Unorm8x4(values) => bytemuck::cast_slice(values),
        }
    }
}

impl IntoBytesMut for VertexAttributeValues {
    fn to_bytes_mut(&mut self) -> &mut [u8] {
        match self {
            Self::Float32(values) => bytemuck::cast_slice_mut(values),
            Self::Sint32(values) => bytemuck::cast_slice_mut(values),
            Self::Uint32(values) => bytemuck::cast_slice_mut(values),
            Self::Float32x2(values) => bytemuck::cast_slice_mut(values),
            Self::Sint32x2(values) => bytemuck::cast_slice_mut(values),
            Self::Uint32x2(values) => bytemuck::cast_slice_mut(values),
            Self::Float32x3(values) => bytemuck::cast_slice_mut(values),
            Self::Sint32x3(values) => bytemuck::cast_slice_mut(values),
            Self::Uint32x3(values) => bytemuck::cast_slice_mut(values),
            Self::Float32x4(values) => bytemuck::cast_slice_mut(values),
            Self::Sint32x4(values) => bytemuck::cast_slice_mut(values),
            Self::Uint32x4(values) => bytemuck::cast_slice_mut(values),
            Self::Sint16x2(values) => bytemuck::cast_slice_mut(values),
            Self::Snorm16x2(values) => bytemuck::cast_slice_mut(values),
            Self::Uint16x2(values) => bytemuck::cast_slice_mut(values),
            Self::Unorm16x2(values) => bytemuck::cast_slice_mut(values),
            Self::Sint16x4(values) => bytemuck::cast_slice_mut(values),
            Self::Snorm16x4(values) => bytemuck::cast_slice_mut(values),
            Self::Uint16x4(values) => bytemuck::cast_slice_mut(values),
            Self::Unorm16x4(values) => bytemuck::cast_slice_mut(values),
            Self::Sint8x2(values) => bytemuck::cast_slice_mut(values),
            Self::Snorm8x2(values) => bytemuck::cast_slice_mut(values),
            Self::Uint8x2(values) => bytemuck::cast_slice_mut(values),
            Self::Unorm8x2(values) => bytemuck::cast_slice_mut(values),
            Self::Sint8x4(values) => bytemuck::cast_slice_mut(values),
            Self::Snorm8x4(values) => bytemuck::cast_slice_mut(values),
            Self::Uint8x4(values) => bytemuck::cast_slice_mut(values),
            Self::Unorm8x4(values) => bytemuck::cast_slice_mut(values),
        }
    }
}
