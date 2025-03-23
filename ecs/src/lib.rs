// Inspired by Bevy's ECS (MIT/Apache-2.0)
// Though this is a very naive first attempt

use ahash::{HashMap, HashSet};

use std::{
    any::{Any, TypeId},
    fmt::{self, Debug, Formatter},
    hash::Hash,
    ops::Deref,
    sync::{Arc, Mutex},
};

#[derive(Debug, Default)]
pub struct World {
    entities: HashMap<EntityId, HashMap<TypeId, Box<dyn Component>>>,
    systems: HashMap<Schedule, HashMap<TypeId, Arc<Mutex<System>>>>,
    resources: HashMap<TypeId, Box<dyn Any>>,
    entity_id_generator: IdGenerator,
}

impl World {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn run_schedule(&mut self, schedule: Schedule) {
        if let Some(systems) = self.systems.get(&schedule) {
            let systems: Vec<_> = systems.values().cloned().collect();
            for system in systems {
                let mut system = system.lock().unwrap();
                system.call(self);
            }
        }
    }

    pub fn spawn(&mut self, components: Vec<Box<dyn Component>>) {
        self.entities.insert(
            EntityId(self.entity_id_generator.generate()),
            components
                .into_iter()
                .map(|c| ((*c).type_id(), c))
                .collect(),
        );
    }

    pub fn insert_resource<R: 'static + Resource>(&mut self, resource: R) {
        self.resources.insert(
            TypeId::of::<R>(),
            Box::new(Arc::new(Mutex::new(Box::new(resource)))),
        );
    }

    pub fn insert_systems(&mut self, schedule: Schedule, systems: Vec<System>) {
        let systems = systems
            .into_iter()
            .map(|sys| (sys.type_id(), Arc::new(Mutex::new(sys))))
            .collect();
        self.systems.insert(schedule, systems);
    }

    pub fn get_entity_commands(&mut self, entity: EntityId) -> Option<EntityCommands> {
        if self.entities.contains_key(&entity) {
            Some(EntityCommands {
                entity,
                world: self,
            })
        } else {
            None
        }
    }

    pub fn get<P: SystemParam>(&self) -> Option<P> {
        P::get_from_world(self)
    }
}

pub struct EntityCommands<'w> {
    entity: EntityId,
    world: &'w mut World,
}

impl EntityCommands<'_> {
    pub fn insert(&mut self, components: Vec<Box<dyn Component>>) {
        self.world
            .entities
            .get_mut(&self.entity)
            .unwrap()
            .extend(components.into_iter().map(|c| ((*c).type_id(), c)));
    }

    pub fn get<C: Component + 'static>(&self) -> Option<&C> {
        self.world
            .entities
            .get(&self.entity)?
            .get(&TypeId::of::<C>())?
            .as_any()
            .downcast_ref::<C>()
    }

    pub fn remove(&mut self) {
        self.world.entities.remove(&self.entity);
    }
}

pub trait Component: Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Debug + Send + Sync + 'static> Component for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl PartialEq for dyn Component {
    fn eq(&self, other: &Self) -> bool {
        self.type_id() == other.type_id()
    }
}

impl Eq for dyn Component {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EntityId(u32);

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

pub struct System(pub Box<dyn FnMut(&mut World)>);

unsafe impl Send for System {}
unsafe impl Sync for System {}

impl System {
    pub fn call(&mut self, world: &mut World) {
        (self.0)(world);
    }
}

impl Debug for System {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "System")
    }
}

pub trait SystemParam: Debug {
    fn get_from_world(world: &World) -> Option<Self>
    where
        Self: Sized;
}

#[derive(Debug, Clone)]
pub struct Res<R: Resource>(Arc<R>);

impl<R: Resource> Deref for Res<R> {
    type Target = R;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref()
    }
}

#[derive(Debug, Clone)]
pub struct ResMut<R: Resource>(pub Arc<Mutex<R>>);

pub trait Resource: Debug + Send + Sync {}

impl<R: Resource + 'static> SystemParam for Res<R> {
    fn get_from_world(world: &World) -> Option<Self> {
        world
            .resources
            .get(&TypeId::of::<R>())?
            .downcast_ref::<Arc<R>>()
            .cloned()
            .map(Res)
    }
}

impl<R: Resource + 'static> SystemParam for ResMut<R> {
    fn get_from_world(world: &World) -> Option<Self> {
        world
            .resources
            .get(&TypeId::of::<R>())?
            .downcast_ref::<Arc<Mutex<R>>>()
            .cloned()
            .map(ResMut)
    }
}

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

#[allow(dead_code)]
mod tests {
    use super::*;
    #[test]
    fn basic_ecs_test() {
        let mut world = World::new();
        world.insert_systems(Schedule::Startup, vec![System(Box::new(system))]);
        world.insert_resource(Person { name: "Anthony" });
        world.run_schedule(Schedule::Startup);
    }

    fn system(world: &mut World) {
        if let Some(person) = world.get::<Res<Person>>() {
            println!("person: {:?}", person);
        } else {
            println!("Person not found!");
        }
    }

    #[derive(Debug)]
    struct Person {
        name: &'static str,
    }

    impl Resource for Person {}
}
