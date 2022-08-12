#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
    Get words from Wikcionario with information of the Spanish variant they belong to using the MediaWiki API.
    Date: 10.08.2022
    Author: cristinae
    TODO: work with dumps instead of API so that results are reproducible
'''

import re
import requests
import logging
import argparse
import codecs

# these variants are categories in Wikcionario
variants = ['ES:América', 'ES:América_Central','ES:América_del_Norte‎', 'ES:América_del_Sur','ES:Caribe‎', 'ES:España', 'ES:Argentina‎', 'ES:Bolivia', 'ES:Chile', 'ES:Colombia', 'ES:Cono_Sur‎', 'ES:Costa_Rica', 'ES:Cuba', 'ES:Ecuador‎', 'ES:El_Salvador', 'ES:Guatemala', 'ES:Honduras‎', 'ES:México', 'ES:Nicaragua', 'ES:Panamá', 'ES:Paraguay‎', 'ES:Perú', 'ES:Puerto_Rico', 'ES:República_Dominicana', 'ES:Río_de_la_Plata', 'ES:Uruguay','ES:Venezuela'#‎,'ES:Venezuela'
]
# the exceptions are the "Uso" that a word in Wikcionario can have (besides the language variant)
exceptions=['académico','afectuoso','anglonormando','anticuado','coloquial','culto','despectivo','desusado','epiceno','eufemismo','familiar','figurado','formal','haquetía','infantil','informal', 'infrecuente','jerga','jocoso','lenfardismo','lunfardismo','Lunfardismo','len','literario','malsonante','mayúscula','medieval','minúscula', 'neolatín','obsoleto','poético','poco usado','préstamo','regiones','Utrep','rural','se usa', 'se emplea', 'Se usa', 'sutp','suts','umcs','utcs','utcp','úsase']


def get_category_members(category):
	'''
	Use the MediaWiki API to get all category members of a Wiktionary category. 
	Takes a category name. Returns a list of pagetitles.
	Adapted from:
	https://github.com/hslh/pie-detection/blob/master/wiktionary.py
	'''

	titles = []
	cont = True
	cmcontinue = '' # Continuation point for query
	# Get titles until no members left
	while(cont):
		# Construct query
		endpoint = 'https://es.wiktionary.org/w/api.php?' # Wiktionary API endpoint
		action = 'action=' + 'query' # Which action to take (query, naturally)
		format = 'format=' + 'json' # Output format
		lists = 'list=' + 'categorymembers'
		cmtitle = 'cmtitle=Categoría:' + category
		cmtitle = re.sub(' ', '%20', cmtitle)
		cmlimit = 'cmlimit=' + '500' # Query result limit
		cmprop = 'cmprop=' + 'title' # Get page titles only
		
		query = endpoint + '&'.join([action, format, lists, cmtitle, cmprop, cmlimit])
		if cmcontinue: # Adding cmcontinue to query makes sure it continues from end of previous query
			query += '&cmcontinue=' + cmcontinue

		# Get and process results
		res_json = requests.get(query).json()
		# Collect page titles
		category_members = res_json['query']['categorymembers']
		for category_member in category_members:
			title = category_member['title']
			if not re.search('(^Appendix:)|(^Categoría:)', title): # Filter out special pages 
			  titles.append(title.strip())
		# Check for more members in category
		try:
			cmcontinue = res_json['continue']['cmcontinue']
			cont = True
		except KeyError:
			cont = False

	return sorted(list(set(titles)))


def get_page(title):
	'''
	Extraction of the categories related to the language, the language variant and the synonyms
	for each word sense in the Wikcionario page "title". The method uses the MediaWiki API.
	'''
	# Construct query
        # https://es.wiktionary.org/w/api.php?action=parse&page=cajeta&prop=wikitext|categories
	endpoint = 'http://es.wiktionary.org/w/api.php?' # Wiktionary API endpoint
	action = 'action=' + 'parse' # Which action to take
	format = 'format=' + 'json' # Output format
	prop = 'prop=' + 'wikitext|categories' # What info to get
	page = 'page=' + title
	page = re.sub(' ', '%20', page)
	query = endpoint + '&'.join([action, format, prop, page])

	# Process result
	try:
		#print(query)
		result = requests.get(query).json()
		# Let's extract the categories in the page
		temp1 = result['parse']['categories'] 
		cats = ''
		for elem in temp1:
			if 'hidden' not in elem and re.search('ES:', elem['*']):
			   p = re.search('(ES:.*)', elem['*'])
			   if(p[0] in variants):
			      cats = cats+p[0]+'|'
			      
		# Let's extract the country from templates in the raw wikitext
		# they might be the same as the categories or not
		temp2 = result['parse']['wikitext'] 
		text = temp2['*'].split('\n')
		#print(text)
		sense=0
		ignore=0
		ambito=''
		sinonimos=''
		term=''
		wordSenses = set()
		for line in text:
			if(line.startswith(';') or line.startswith('=')): 
			   if not ignore:
			      if(cats==''): cats='-'
			      if(ambito=='' or '|nota' in ambito  or '|alt' in ambito): ambito='-'
			      if(sinonimos=='' or '|nota' in sinonimos or '|alt' in sinonimos or 'leng=' in sinonimos): sinonimos='-'
			      term = title + ',' + cats.rstrip('|') + ',' +ambito.rstrip().lstrip() + ',' + sinonimos.rstrip().lstrip().rstrip('|')
			      #print(term)
			      wordSenses.add(term)
			   else:
			      ignore = 0
			   ambito = '-'
			   sinonimos = ''
			   term = ''
			   sense = sense+1
			elif(sense>0):
		# Different notations are used:
		#:*'''Ámbito:''' {{América Central|México}}
		#:*'''Ámbito:''' Colombia
		#:*'''Ámbito:''' [[Chile]], [[Perú]], [[Bolivia]], [[Ecuador]], América Central"
		#{{ámbito|Venezuela}}
		#{{ámbito|Bolivia|Chile|México}}.<ref name=drae>{{DRAE}}</ref>
		#{{uso|coloquial}}
                #{{uso|Colombia|Cuba|Panamá|Perú|España|Venezuela}}
			   if(any(line.lower().startswith('{{uso|'+elem) for elem in exceptions)):
			     ignore=1
			     continue
			   if(line.startswith('{{uso|')):
			     p = re.search('{{uso\|(.*?)}}',line)
			     ambito=p[1]
			   elif(line.startswith(':*\'\'\'Ámbito:')):
			     p = re.search('{{(.*?)}}',line)
			     if p is None:
			        p = re.search(':*\'\'\'Ámbito:\'\'\'(.*)',line)
			     ambito=p[1]  
			   elif(line.startswith('{{ámbito|')):
			     p = re.search('{{ámbito\|(.*?)}}',line)
			     ambito=p[1]
		# Notes can be in the middle of the string	     
                #{{sinónimo|dulce de leche|manjar blanco}}.
                #{{sinónimo|durazno}}.
                #{{sinónimo|cura|palta|nota2=Argentina, Bolivia, Chile, Perú, Uruguay|palto}}
			   elif(line.startswith('{{sinónimo|')):
			     p = re.search('{{sinónimo\|(.*?)}}',line)
			     p2 = p[0].split('|')
			     for token in p2:
			         if not (token.startswith('nota') or token.startswith('{{sinónimo')):
			            sinonimos=sinonimos+token+'|'
			     sinonimos=sinonimos.replace('}}|','')
			ambito=ambito.replace(']], ','|')
			ambito=ambito.replace('[','')
			ambito=ambito.replace(',','|')
		# The last item was not printed
		if not ignore:
			if(cats==''): cats='-'
			if(ambito=='' or '|nota' in ambito  or '|alt' in ambito): ambito='-'
			if(sinonimos=='' or '|nota' in sinonimos or '|alt' in sinonimos or 'leng=' in sinonimos): sinonimos='-'
			term = title + ',' + cats.rstrip('|') + ',' +ambito.rstrip().lstrip() + ',' + sinonimos.rstrip().lstrip().rstrip('|')
			#print(term)
			wordSenses.add(term)

	except KeyError:
		return 
		
	return wordSenses
	

def main():
	parser = argparse.ArgumentParser(description="Running WikcionarioVariants")
	parser.add_argument("--category", type=str, default=None, help="category to extract words from", required=True)
	parser.add_argument("--oFile", type=str, default=None, help="Output file (csv)")
	args = parser.parse_args()

	logging.basicConfig(level=logging.INFO)
	logging.info("Retrieving pages in "+ args.category)    
	pages = get_category_members(args.category)
	#pages=['hacer pellas']

	logging.info("Creating output file with content 'word, Wikcionario category, ambito or uso, synonyms'") 
	if(args.oFile):   
		with codecs.open(args.oFile, "w", "utf8") as f:
			f.write('word, Wikcionario category, ambito or uso, synonyms\n')
			for page in pages:
				wordSenses = get_page(page)
				for instance in wordSenses:
			   	    f.write(instance)
			   	    f.write('\n')
			f.close()
	else:
		print('word, Wikcionario category, ambito or uso, synonyms')
		for page in pages:
			wordSenses = get_page(page)
			for instance in wordSenses:
 			    print(instance)
    
if __name__ == "__main__":
  main()
  
