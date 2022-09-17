# -*- coding: utf-8 -*-
# @Author : Zip
# @Moto   : Knowledge comes from decomposition
from __future__ import absolute_import, division, print_function
"""Registry utility."""
from absl import logging


def register(registered_collection, reg_key):
    """Register decorated function or class to collection."""
    def decorator(fn_or_cls):
        """Put fn_or_cls in the dictionary."""
        if isinstance(reg_key, str):
            hierarchy = reg_key.split("/")
            collection = registered_collection
            for h_idx, entry_name in enumerate(hierarchy[:-1]):
                if entry_name not in collection:
                    collection[entry_name] = {}
                collection = collection[entry_name]
                if not isinstance(collection, dict):
                    raise KeyError(
                        "Collection path {} at position {} already registered as "
                        "a function or class.".format(entry_name, h_idx))
            leaf_reg_key = hierarchy[-1]
        else:
            collection = registered_collection
            leaf_reg_key = reg_key

        if leaf_reg_key in collection:
            if "beta" in fn_or_cls.__module__:
                # TODO(yeqing): Clean this temporary branch for beta.
                logging.warn(
                    "Duplicate registeration of beta module "
                    "name %r new %r old %r", reg_key, collection[leaf_reg_key],
                    fn_or_cls.__module__)
                return fn_or_cls
            else:
                raise KeyError(
                    "Function or class {} registered multiple times.".format(
                        leaf_reg_key))

        collection[leaf_reg_key] = fn_or_cls
        return fn_or_cls

    return decorator


def lookup(registered_collection, reg_key):
    """Lookup and return decorated function or class in the collection."""
    if isinstance(reg_key, str):
        hierarchy = reg_key.split("/")
        collection = registered_collection
        for h_idx, entry_name in enumerate(hierarchy):
            if entry_name not in collection:
                raise LookupError(
                    f"collection path {entry_name} at position {h_idx} is never "
                    f"registered. Please make sure the {entry_name} and its library is "
                    "imported and linked to the trainer binary.")
            collection = collection[entry_name]
        return collection
    else:
        if reg_key not in registered_collection:
            raise LookupError(
                f"registration key {reg_key} is never "
                f"registered. Please make sure the {reg_key} and its library is "
                "imported and linked to the trainer binary.")
        return registered_collection[reg_key]
