import pickle
import random
import os

def get_group2cnames():
	return {
		0:	["tunicate_partial", "siphonophore_partial", "hydromedusae_partial_dark"],
		1:	["acantharia_protist", "acantharia_protist_big_center",
			"acantharia_protist_halo", "trochophore_larvae", "hydromedusae_aglaura",
			"hydromedusae_narco_young"],
		2:	["trichodesmium_tuft", "trichodesmium_bowtie", "trichodesmium_puff", 
			"trichodesmium_multiple", "protist_dark_center"],
		3:	["artifacts", "artifacts_edge"],
		4:	["fish_larvae_deep_body", "fish_larvae_leptocephali", "fish_larvae_medium_body",
			"fish_larvae_myctophids", "fish_larvae_thin_body", "fish_larvae_very_thin_body",
			"crustacean_other", "pteropod_triangle", "amphipods", "chordate_type1"],
		5:	["copepod_calanoid", "copepod_calanoid_eggs", "copepod_calanoid_eucalanus",
			"copepod_calanoid_flatheads", "copepod_calanoid_frillyAntennae",
			"copepod_calanoid_large", "copepod_calanoid_large_side_antennatucked",
			"copepod_calanoid_octomoms", "copepod_calanoid_small_longantennae",
			"copepod_cyclopoid_oithona", "copepod_cyclopoid_oithona_eggs", "copepod_other",
			"detritus_other"],
		6:	["stomatopod", "amphipods", "shrimp-like_other", "shrimp_caridean", "decapods",
			"shrimp_sergestidae", "shrimp_zoea", "euphausiids", "euphausiids_young"],
		7:	["chaetognath_non_sagitta", "chaetognath_other", "chaetognath_sagitta"],
		8:	["echinopluteus","echinoderm_larva_pluteus_early", "echinoderm_larva_pluteus_urchin",
			 "echinoderm_larva_pluteus_brittlestar", "echinoderm_larva_pluteus_typeC"],
		9:	["echinoderm_larva_seastar_bipinnaria", "echinoderm_larva_seastar_brachiolaria"],
		10: ["appendicularian_fritillaridae", "appendicularian_s_shape",
			"appendicularian_slight_curve", "appendicularian_straight"],
		11: ["hydromedusae_haliscera", "hydromedusae_haliscera_small_sideview",
			"hydromedusae_shapeA", "hydromedusae_typeE", "hydromedusae_typeD", 
			"hydromedusae_sideview_big", "hydromedusae_other"],
		12: ["jellies_tentacles","pteropod_butterfly", "pteropod_theco_dev_seq",
			"hydromedusae_liriope", "hydromedusae_narcomedusae", "hydromedusae_narco_dark",
			"hydromedusae_solmundella", "hydromedusae_solmaris", "hydromedusae_h15",
			"hydromedusae_shapeB", "hydromedusae_typeD_bell_and_tentacles",
			"hydromedusae_bell_and_tentacles"],
		13:	["tunicate_doliolid_nurse", "hydromedusae_shapeA_sideview_small",
			"hydromedusae_other", "ctenophore_cydippid_no_tentacles",
			"ctenophore_cydippid_tentacles"],
		14: ["detritus_other", "invertebrate_larvae_other_A", "siphonophore_calycophoran_abylidae",
			"siphonophore_other_parts", "protist_star", "echinoderm_seacucumber_auricularia_larva"],
		15: ["siphonophore_calycophoran_sphaeronectes",
			"siphonophore_calycophoran_sphaeronectes_stem",
			"siphonophore_calycophoran_sphaeronectes_young"],
		16:	["diatom_chain_string", "diatom_chain_tube", "jellies_tentacles", "polychaete",
			"heteropod","fecal_pellet", "detritus_filamentous", "invertebrate_larvae_other_B",
			"tunicate_doliolid_nurse", "tunicate_salp_chains", "siphonophore_other_parts",
			"siphonophore_physonect", "siphonophore_physonect_young", "unknown_sticks",
			"siphonophore_calycophoran_rocketship_adult", "radiolarian_colony",
			"siphonophore_calycophoran_rocketship_young", "radiolarian_chain", "ctenophore_cestid"],
		17: ["ephyra", "copepod_cyclopoid_copilia","pteropod_butterfly", 
			"tornaria_acorn_worm_larvae", "tunicate_salp", "ctenophore_lobate"],
		18: ["protist_fuzzy_olive", "protist_noctiluca", "protist_other", 
			"radiolarian_colony", "detritus_other", "tunicate_doliolid"],
		19: ["detritus_blob", "hydromedusae_typeF", "protist_dark_center"]
		}
path_class_names = "../data/train_feature_extraction/class_names.dat"
class_names = pickle.load(open(path_class_names, "rb") )
path_classes = "../data/train_feature_extraction/classes.dat"
classes = pickle.load(open(path_classes, "rb") )
def get_cname2class():
	return {class_names[c]: c for c in class_names}
def get_group2class():
	return {gid: list(map(lambda n: get_cname2class()[n], cnames))
					for gid, cnames in get_group2cnames().items()}
def get_class2group():
	class2group = {c: [] for c in classes}
	for gid, clist in get_group2class().items():
		for c in clist:
			class2group[c] += [gid]
	return class2group
# print(group2class)
# class2group = {}
# for group_id in group2class:
# 	group_list = class_name[group]
# 	for c_name in group_list:
# 		class2group[c] = group_id
# for n in sorted(name2class):
# 	print(n)