import os 
import gc
import torch
import numpy as np
from tqdm import tqdm
from Bio import pairwise2
from Bio.PDB import PDBIO
from Bio.PDB import MMCIFParser, PDBParser
from transformers import T5Tokenizer, T5EncoderModel

from amino_acid import *


DICT={'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
        'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
        'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
        'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}


def parse_complex(pdb_file, chains1, chains2):
    if pdb_file.lower().endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(PERMISSIVE=1, QUIET=True)
        
    model = parser.get_structure("X", pdb_file)[0] 

    seqq1, seqq2, chain_annotation = '', '', ''
    for chain1 in chains1:
        if chain1 in model:
            chain = model[chain1]
            for residue in chain:
                if residue.get_resname() in DICT: seqq1 += DICT[residue.get_resname()]
                else: continue
                chain_annotation += chain.get_id()
    for chain2 in chains2:
        if chain2 in model:
            chain = model[chain2]
            for residue in chain:
                if residue.get_resname() in DICT: seqq2 += DICT[residue.get_resname()]
                else: continue
                chain_annotation += chain.get_id()

    return seqq1, seqq2, chain_annotation 


def get_chain_coord(chain):
    chain_X, chain_aa = [], ""
    for residue in chain:
        if residue.get_resname() in DICT: chain_aa += DICT[residue.get_resname()]
        else: continue
        
        atom_coord_list = []
        for atom in ["N", "CA", "C", "O"]:
            try:
                coord = residue[atom].get_vector().get_array().astype(np.float32)
            except:
                coord = residue.center_of_mass()
            atom_coord_list.append(coord)
        R_group = []
        for atom in residue:
            if atom.id not in ["N", "CA", "C", "O", "H"]:
                R_group.append(atom.get_vector().get_array().astype(np.float32))
        if R_group == []:
            R_group.append(atom_coord_list[1]) 
        atom_coord_list.append(np.array(R_group).mean(0))
        
        chain_X.append(atom_coord_list)
    return chain_X, chain_aa

def get_pdb_coord(pdb_file, chains1, chains2):
    if pdb_file.lower().endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(PERMISSIVE=1, QUIET=True)
    
    model = parser.get_structure('struct', pdb_file)[0]
    X, aa1, aa2 = [], "", ""
    
    for chain1 in chains1:
        if chain1 in model:
            chain = model[chain1]
            chain_X, chain_aa = get_chain_coord(chain)
            X.extend(chain_X)
            aa1 += chain_aa
    for chain2 in chains2:
        if chain2 in model:
            chain = model[chain2]
            chain_X, chain_aa = get_chain_coord(chain)
            X.extend(chain_X)
            aa2 += chain_aa

    return np.array(X), aa1, aa2


def process_dssp(dssp_file):
    aa_type = "ACDEFGHIKLMNPQRSTVWY" # amino acid type
    SS_type = "HBEGITSC" # secondary structure type
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230] # standard value of relative solvent accessibility

    with open(dssp_file, "r") as f:
        lines = f.readlines()
    seq = ""
    dssp_feature = []
    
    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*":  continue
        seq += aa
        SS = lines[i][16]
        if SS == " ":   SS = "C"
        SS_vec = np.zeros(8)
        SS_vec[SS_type.find(SS)] = 1
        ACC = float(lines[i][34:38].strip())
        ASA = min(1, ACC / rASA_std[aa_type.find(aa)])
        dssp_feature.append(np.concatenate((np.array([ASA]), SS_vec)))

    return seq, dssp_feature

def match_dssp(seq, dssp, ref_seq):
    alignments = pairwise2.align.globalxx(ref_seq, seq)
    ref_seq = alignments[0].seqA
    seq = alignments[0].seqB 

    padded_item = np.zeros(9)
    new_dssp = []
    for aa in seq:
        if aa == "-":   new_dssp.append(padded_item)
        else:   new_dssp.append(dssp.pop(0))

    matched_dssp = []
    for i in range(len(ref_seq)):
        if ref_seq[i] == "-":
            continue
        matched_dssp.append(new_dssp[i])

    return matched_dssp

def get_chain_dssp(ID, ref_seq, dssp_path):
    os.system("./models/mkdssp -i {}/{}.pdb -o {}/{}.dssp".format(dssp_path, ID, dssp_path, ID))

    dssp_seq, dssp_matrix = process_dssp("{}/{}.dssp".format(dssp_path, ID))

    if dssp_seq != ref_seq:
        dssp_matrix = match_dssp(dssp_seq, dssp_matrix, ref_seq)

    assert (len(dssp_matrix)==len(ref_seq))
    os.system("rm {}/{}.dssp".format(dssp_path, ID))
    os.system("rm {}/{}.pdb".format(dssp_path, ID))
    
    return dssp_matrix

def get_dssp(ID, pdb_file, chains1, chains2, dssp_path):
    if pdb_file.lower().endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(PERMISSIVE=1, QUIET=True)
    structure = parser.get_structure('temp', pdb_file)
    
    dssp_tensor = []
    for chain1 in chains1:
        new_structure = structure.copy()
        tmp_seq = ""
        for chain in list(new_structure[0]):
            if chain.id != chain1: new_structure[0].detach_child(chain.id)
            else:
                for residue in chain: tmp_seq += DICT[residue.get_resname()]
        io = PDBIO()
        io.set_structure(new_structure)
        io.save(f"{dssp_path}/{ID}_{chain1}.pdb")
        dssp_tensor.extend(get_chain_dssp(f"{ID}_{chain1}", tmp_seq, dssp_path))
    
    for chain2 in chains2:
        new_structure = structure.copy()
        tmp_seq = ""
        for chain in list(new_structure[0]):
            if chain.id != chain2: new_structure[0].detach_child(chain.id)
            else:
                for residue in chain: tmp_seq += DICT[residue.get_resname()]
        io = PDBIO()
        io.set_structure(new_structure)
        io.save(f"{dssp_path}/{ID}_{chain2}.pdb")
        dssp_tensor.extend(get_chain_dssp(f"{ID}_{chain2}", tmp_seq, dssp_path))
    
    return np.array(dssp_tensor)


def get_node_feature(recseq, ligseq):
    allseq = recseq + ligseq
    node_feat = []
    for i in range(len(allseq)):
        # aa class
        aa_class = amino_acids_by_letter[allseq[i]]
        # res_type+entity
        feat = list(aa_class.onehot)
        if i < len(recseq): feat += [0]
        else: feat += [1]
        node_feat.append(feat)
    return np.array(node_feat)


def get_ProtTrans(ID_list, seq_list, Min_protrans, Max_protrans, ProtTrans_path, outpath, gpu):
    # Load the vocabulary and ProtT5-XL-UniRef50 Model
    tokenizer = T5Tokenizer.from_pretrained(ProtTrans_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(ProtTrans_path)
    gc.collect()

    # Load the model into the GPU if avilabile and switch to inference mode
    device = torch.device('cuda:' + gpu if torch.cuda.is_available() and gpu else 'cpu')
    model = model.to(device)
    model = model.eval()

    print("Extracting ProtTrans embeddings...")
    for i in tqdm(range(len(ID_list))):
        batch_ID_list = [ID_list[i]] # batch size = 1
        batch_seq_list = [" ".join(list(seq_list[i]))]

        ids = tokenizer.batch_encode_plus(batch_seq_list, add_special_tokens=True, padding=True)
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)
        embedding = embedding.last_hidden_state.cpu()

        for seq_num in range(len(embedding)):
            seq_len = (attention_mask[seq_num] == 1).sum()
            seq_emd = embedding[seq_num][:seq_len-1]
            seq_emd = (seq_emd - Min_protrans) / (Max_protrans - Min_protrans) 
            torch.save(seq_emd, outpath + batch_ID_list[seq_num] + '.tensor')