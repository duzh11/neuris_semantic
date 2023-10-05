import json
import os
import numpy as np
import csv

def load_scannet_label_mapping(path):
    """ Returns a dict mapping scannet category label strings to scannet Ids

    scene****_**.aggregation.json contains the category labels as strings 
    so this maps the strings to the integer scannet Id

    Args:
        path: Path to the original scannet data.
              This is used to get scannetv2-labels.combined.tsv

    Returns:
        mapping: A dict from strings to ints
            example:
                {'wall': 1,
                 'chair: 2,
                 'books': 22}

    """

    mapping = {}
    with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for i, line in enumerate(tsvreader):
            if i==0:
                continue
            scannet_id, name = int(line[0]), line[1]
            mapping[name] = scannet_id

    return mapping


def load_scannet_nyu3_mapping(path,manhattan=False):
    """ Returns a dict mapping scannet Ids to NYU40 Ids

    Args:
        path: Path to the original scannet data. 
            This is used to get scannetv2-labels.combined.tsv

    Returns:
        mapping: A dict from ints to ints
            example:
                {1: 1,
                 2: 5,
                 22: 23}

    """
    mapping = {}
    with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for i, line in enumerate(tsvreader):
            if i==0:
                continue
            scannet_id, nyu40id, nyu40class = int(line[0]), int(line[4]), line[7]
            if nyu40class=='wall':
                mapping[scannet_id] = nyu40id
            elif nyu40class=='floor':
                mapping[scannet_id] = nyu40id
            else:
                mapping[scannet_id] = 0
            
            if manhattan:
                if nyu40class=='door':
                    mapping[scannet_id] = 1 # regard door as wall
                if nyu40class=='whiteboard':
                    mapping[scannet_id] = 1 # regard white board as wall
                if nyu40class=='floor mat':
                    mapping[scannet_id] = 2 # regard floor mat as floor

    return mapping


def load_scannet_nyu40_mapping(path, manhattan=False):
    """ Returns a dict mapping scannet Ids to NYU40 Ids

    Args:
        path: Path to the original scannet data. 
            This is used to get scannetv2-labels.combined.tsv

    Returns:
        mapping: A dict from ints to ints
            example:
                {1: 1,
                 2: 5,
                 22: 23}

    """

    mapping = {}
    with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for i, line in enumerate(tsvreader):
            if i==0:
                continue
            scannet_id, nyu40id, nyu40class = int(line[0]), int(line[4]),line[7]
            mapping[scannet_id] = nyu40id
            
            if manhattan:
                if nyu40class=='door':
                    mapping[scannet_id] = 1 # regard door as wall
                if nyu40class=='whiteboard':
                    mapping[scannet_id] = 1 # regard white board as wall
                if nyu40class=='floor mat':
                    mapping[scannet_id] = 2 # regard floor mat as floor
    
    return mapping


def load_scannet_nyu13_mapping(path, manhattan=False):
    """ Returns a dict mapping scannet Ids to NYU40 Ids

    Args:
        path: Path to the original scannet data. 
            This is used to get scannetv2-labels.combined.tsv

    Returns:
        mapping: A dict from ints to ints
            example:
                {1: 1,
                 2: 5,
                 22: 23}

    """

    mapping = {}
    with open(os.path.join(path, 'scannetv2-labels.combined.tsv')) as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter='\t')
        for i, line in enumerate(tsvreader):
            if i==0:
                continue
            scannet_id, nyu13id, nyu40class= int(line[0]), int(line[5]), line[7]
            mapping[scannet_id] = nyu13id

            if manhattan:
                if nyu40class=='door':
                    mapping[scannet_id] = 12 # regard door as wall
                if nyu40class=='whiteboard':
                    mapping[scannet_id] = 12 # regard white board as wall
                if nyu40class=='floor mat':
                    mapping[scannet_id] = 5 # regard floor mat as floor    
    
    return mapping