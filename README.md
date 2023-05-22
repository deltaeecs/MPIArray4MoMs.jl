# MPIArray4MoMs

![star](https://img.shields.io/github/stars/deltaeecs/MPIArray4MoMs.jl?style=social)

[![Build Status](https://github.com/deltaeecs/MPIArray4MoMs.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/deltaeecs/MPIArray4MoMs.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/deltaeecs/MPIArray4MoMs.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/deltaeecs/MPIArray4MoMs.jl)

![Size](https://img.shields.io/github/repo-size/deltaeecs/MPIArray4MoMs.jl)
![Downloads](https://img.shields.io/github/downloads/deltaeecs/MPIArray4MoMs.jl/total)
![License](https://img.shields.io/github/license/deltaeecs/MPIArray4MoMs.jl)

## 介绍

提供 CEM\_MoMs ([![github](https://img.shields.io/badge/github-blue.svg)](https://github.com/deltaeecs/CEM_MoMs.jl), [![gitee](https://img.shields.io/badge/gitee-red.svg)](https://gitee.com/deltaeecs/CEM_MoMs.jl)) 中 [![MoM_MPI](https://img.shields.io/badge/MoM__MPI-yellow.svg)](https://github.com/deltaeecs/MoM_MPI.jl) 包的 MPI 数组实现。主要提供了MPI 数组类型、实现 MPI 数组在进程间的通信、MPI 矩阵与向量乘积等功能、MLFMA 中的层内辐射、配置函数的数据传输等。