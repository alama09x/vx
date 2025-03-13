use std::marker::PhantomData;

use ahash::HashMap;

use crate::{component::Component, schedule::Schedule, system::System};

type Entity = u32;

pub struct Manager {
    entities: HashMap<Entity, Vec<Box<dyn Component>>>,
    systems: HashMap<Schedule, System>,
    schedule: Schedule,
}

pub struct Query<D, F = ()> {
    phantom_data: PhantomData<(D, F)>,
}
