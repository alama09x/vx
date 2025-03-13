use thiserror::Error;

use crate::{component::Resource, manager::Query};

#[derive(Error, Debug)]
enum SystemError {}

pub struct SystemRequest<D, F = ()> {
    queries: Vec<Box<Query<D, F>>>,
    resources: Vec<Box<dyn Resource>>,
}

pub struct System(pub Box<dyn Fn(dyn SystemRequest) -> Result<(), SystemError>>);
