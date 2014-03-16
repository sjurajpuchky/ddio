ddio
====
Direct device input output

> Time to time whe need to access directly other devices as memory, keyboard, cpu and emulated devices like host functions input/output from any device like GPU grid, cluster node another CPU, CPU extension cards like Tegra or other which are out of the context. By software bridge which, emulates DMA/RDMA, MMU/RMMU, IO/RIO and FUNC/HFUNC techniques on abstract emulated/virtualized hardware layer.

Common use cases
================
Ddio technique can be used in following 
- call host function from CUDA device
- access other device from CUDA device function
- *

Requirements
============
* MMU
* CUDA with compute engine at least 2.0 and 
  CUDA implementation at least 5.0 with RDMA support
* DMA


Specification is in progress...
