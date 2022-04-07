from . import l1_pruner, snip_pruner

pruner_dict = {
    'l1': l1_pruner,
    't_snip': snip_pruner,
    's_snip': snip_pruner,
}