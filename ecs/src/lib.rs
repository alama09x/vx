use ahash::{HashMap, HashSet};

use std::{
    any::{Any, TypeId},
    fmt::{self, Debug, Formatter},
    hash::Hash,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Entity(pub u32);

#[derive(Debug, Default)]
pub struct World {
    entities: HashMap<Entity, HashMap<ComponentId, Box<dyn Component>>>,
    systems: HashMap<Schedule, HashMap<SystemId, System>>,
    resources: HashMap<ResourceId, Resource>,
    entity_id_generator: IdGenerator,
}

impl World {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn spawn(&mut self, components: Vec<Box<dyn Component>>) {
        self.entities.insert(
            Entity(self.entity_id_generator.generate()),
            components
                .into_iter()
                .map(|c| (ComponentId((*c).type_id()), c))
                .collect(),
        );
    }

    pub fn insert_resource<R: 'static + ResourceTrait>(&mut self, resource: R) {
        self.resources
            .insert(ResourceId(resource.type_id()), Resource(Box::new(resource)));
    }

    pub fn insert_systems(&mut self, schedule: Schedule, systems: Vec<System>) {
        let systems = systems
            .into_iter()
            .map(|sys| (SystemId(sys.type_id()), sys))
            .collect();
        self.systems.insert(schedule, systems);
    }
}

pub struct EntityCommands {
    entity: Entity,
    world: &'static mut World,
}

impl EntityCommands {
    pub fn insert(&mut self, components: Vec<Box<dyn Component>>) {
        self.world.entities.get_mut(&self.entity).unwrap().extend(
            components
                .into_iter()
                .map(|c| (ComponentId((*c).type_id()), c)),
        );
    }
}

pub trait Component: Debug + Send + Sync {}

impl PartialEq for dyn Component {
    fn eq(&self, other: &Self) -> bool {
        self.type_id() == other.type_id()
    }
}

impl Eq for dyn Component {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SystemId(TypeId);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ComponentId(TypeId);

#[derive(Debug, Default)]
pub struct IdGenerator {
    lookup_table: HashSet<u32>,
}

impl IdGenerator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn generate(&mut self) -> u32 {
        fn generate_id() -> u32 {
            rand::random_range(0..=u32::MAX)
        }

        let mut id = generate_id();
        while self.lookup_table.contains(&id) {
            id = generate_id();
        }

        self.lookup_table.insert(id);
        id
    }
}

impl SystemId {}

type SystemParams = Vec<Box<dyn SystemParam>>;
pub struct System {
    params: SystemParams,
    callback: Box<dyn FnMut(&mut SystemParams)>,
}

impl System {
    pub fn call(&mut self) {
        (self.callback)(&mut self.params);
    }
}

impl Debug for System {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "System with params {:?}", self.params)
    }
}

pub trait SystemParam: Debug {}

#[derive(Debug)]
pub struct Resource(pub Box<dyn ResourceTrait>);

pub trait ResourceTrait: Debug + Send + Sync {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceId(TypeId);

#[derive(Debug, PartialEq, Eq, Hash)]
pub enum Schedule {
    Initialize,
    PreStartup,
    Startup,
    PostStartup,
    Update,
    PostUpdate,
    Cleanup,
    Exit,
}
