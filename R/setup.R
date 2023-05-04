# Keras uses a tensorflow backend, and thus the project needs python to run.
# renv::snapshot calls pip/conda on the back end to manage *python* dependencies.
renv::use_python()
