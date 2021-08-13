from math import *
from random import randint
import time
start_time = time.time()
def aodv (hop_count) :

	drop_factor = (1/(hop_count+1)*1.0)
	random_value = randint(0,1)

	if(random_value>drop_factor):
		print("RREQ_packet is forwarded")

	else:
		print("RREQ_packet is dropped")

	return

hop_count = 1 # single-hop

print("For single-hop aodv protocol:\n")

aodv(hop_count);

hop_count = 2 # greater than 1 => multihop

print("For multi-hop aodv protocol:\n")

aodv(hop_count);

print("--- %s seconds ---" % (time.time() - start_time))
