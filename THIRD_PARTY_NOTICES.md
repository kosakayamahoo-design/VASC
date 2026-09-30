# Third-Party Notices

`src/sparse_vggt` derives from FasterVGGT (`brianwang00001/sparse-vggt`) with
research modifications for value-aware selection and cross-layer memory.
Original source headers have been retained. The imported snapshot has no
top-level license file; the upstream repository page inspected during packaging
also did not list one. Distribution permission for those inherited files needs
confirmation before a public release. No blanket license is assigned here.

`third_party/spargeattn` contains the modified SpargeAttn runtime used by the
experiments. Its original Apache-2.0 license and copyright headers are retained.
The release adds a build entry point; CUDA/Python kernel sources are copied
unchanged from the experimental runtime. The modifications predate this
packaging and include support needed by the research scheduler.

VGGT, Pi3, StreamVGGT, DUSt3R, and CroCo are external dependencies. They and
their checkpoints remain subject to their respective upstream terms. No model
weights or datasets are redistributed in this folder.
