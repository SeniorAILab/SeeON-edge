//! Policy thread glue (design §2.3): pose packets into the `seeon-worker`
//! domain, fall and bed accelerator requests, observation coverage (G16) and
//! durable event identity (G20).

pub mod bed;
pub mod bed_exit;
pub mod bed_exit_input;
pub mod coverage;
pub mod emit;
pub mod fall;
pub mod identity;
pub mod incident;
pub mod ingest;
