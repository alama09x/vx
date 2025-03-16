use ahash::HashMap;

use crate::{component::Component, schedule::Schedule, system::System};

pub type Entity = u32;

pub struct Manager {
    entities: HashMap<Entity, Vec<Box<dyn Component>>>,
    systems: HashMap<Schedule, System>,
    schedule: Schedule,
}
