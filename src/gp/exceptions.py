"""
AUTHOR: Ryan Grindle

LAST MODIFIED: Oct 9, 2020

PURPOSE: This file contains some exceptions that can be used
         when k-expressions are used improperly.

NOTES: UnknownPrimitiveError indicates that there is an unknown symbol
       in the k-expression that is known or assumed to be a primitive.

TODO:
"""


class UnknownPrimitiveError(Exception):

    def __init__(self, primitive):
        super().__init__()
        self.primitive = primitive

    def __str__(self):
        return 'ERROR: Missing primitive '+self.primitive+' in primitive2function?'
