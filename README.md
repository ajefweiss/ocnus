# Ocnus - A Magnetic Flux Rope & Solar Wind Modeling Framework

The **ocnus** crate attempts to leverage Rust's type system to provide a flexible framework that aims to simplify the implementation of magnetic flux rope or solar wind models, by making use of commonly shared functionality.

## Core Traits

- [`OcnusCoord`](`crate::coords::OcnusCoords`) for representing generic 3D curvilinear coordinate systems. The trait guarantuees the implementation of coordinate transform functions, in both directions, and the implementation of methods that compute the local co- and contravariant basis vectors. Any coordinate system can be configured using a set of coordinate parameters (static) or using an associated coordinate system state type (non-static). For an extensive description of the implemented coordinate systems, and their associated state types, see [`here`](`crate::coords`).

- [`OcnusObser`](`crate::obser::OcnusObser`) for types that represent a generic observation or measurement. Spacecraft observation time-series, including ephemeris or pointing data, are represented by [`ScObsSeries`](`crate::obser::ScObsSeries`). These series can be added to each other, re-sorted in time by the observation timestamp and also split back into the original components. This allows for an observation scenario with multiple spacecraft to be re-structed as if it were only from one. For a list of implemented observation types see [`here`](`crate::obser`).

- [`OcnusProDeF`](crate::prodef::OcnusProDeF) for types that represent a generic probability density functions (PDFs). This trait guarantuees the implementation of sampling functionality and the calculation of an un-normalized relative density value. For a detailed explanation of the primarily three implemented PDF types see [`here`](`crate::prodef`).