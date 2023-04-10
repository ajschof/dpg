from Bio import SeqIO

with open("opuntia.aln") as handle:
    for record in SeqIO.parse(handle, "clustal"):
        print(record.id)
        
