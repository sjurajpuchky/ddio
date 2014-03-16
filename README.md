ddio
====
Direct device input output

> Time to time whe need to access directly other devices as memory, keyboard, cpu and emulated devices like host functions input/output from any device like GPU grid, cluster node another CPU, CPU extension cards like Tegra or other which are out of the standard CPU model context. By software bridge which, emulates techniques on abstract emulated/virtualized hardware layer.

> Double side binary compatible syscall like execution of code which are separed to functions depended on needs like IO/DMA/MMU/FUNC.

Common use cases
================
Ddio technique can be used in following 
- call host function from CUDA device
- access other device from CUDA device function

Requirements
============
* CUDA with compute engine at least 2.0 and 

Plans
=====
- Finish specification
- Prepare interfaces for virtual machine concept
- Prepare interfaces for libdc Distributed compute
- Prepare integration for HotSpot JavaVM to make JVM domesticated on CUDA
- Prepare interfaces for kgpu (separed project)
- Prepare interfaces for gsyscall (separed project)

> Separed projects are depended on research

Contributors
============
Are welcome, let's give me know any time you wish to join



Specification is in progress...
