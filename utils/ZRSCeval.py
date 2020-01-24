import pkg_resources 
from WDE.readers.gold_reader import *
from WDE.readers.disc_reader import *

wrd_path = pkg_resources.resource_filename(
            pkg_resources.Requirement.parse('WDE'),
                        'WDE/share/flickr30k_word_units.wrd')
phn_path = pkg_resources.resource_filename(
            pkg_resources.Requirement.parse('WDE'),
                        'WDE/share/flickr30k_phone_units.phn')

gold = Gold(wrd_path=wrd_path, 
                phn_path=phn_path) 

disc_clsfile = 'tdev2/WDE/share/discovered_words.class'

discovered = Disc(disc_clsfile, gold) 

from WDE.measures.grouping import * 
from WDE.measures.coverage import *
from WDE.measures.boundary import *
from WDE.measures.ned import *
from WDE.measures.token_type import *


grouping = Grouping(discovered)
grouping.compute_grouping()
print('Grouping precision: ', grouping.precision)
print('Grouping recall: ', grouping.recall)
#print('Grouping fscore: ', grouping.fscore)

coverage = Coverage(gold, discovered)
coverage.compute_coverage()
print('Coverage: ', coverage.coverage)

boundary = Boundary(gold, discovered)
boundary.compute_boundary()
print('Boundary precision: ', boundary.precision)
print('Boundary recall: ', boundary.recall)
#print('Boundary fscore: ', boundary.fscore)

ned = Ned(discovered)
ned.compute_ned()
print('NED: ', ned.ned)

token_type = TokenType(gold, discovered)
token_type.compute_token_type()
print('Token type precision: ', token_type.precision)
print('Token type recall: ', token_type.recall)
#print('Token type fscore: ', token_type.fscore)
