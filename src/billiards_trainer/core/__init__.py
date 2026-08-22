"""Foundation layer: the vocabulary every subsystem shares.

A Track, a BallClass, a TableModel are not vision's private property —
events, the detector strategies and the UI all speak them. While they
lived inside vision/, any subsystem needing the vocabulary had to import
vision, and vision imported those subsystems back: the system review's
L1 pass reported events<->vision and detector_strategies<->vision
cycles. Moving the vocabulary DOWN to its own layer breaks both, and
gives new features an obvious place to find the shared nouns.
"""
