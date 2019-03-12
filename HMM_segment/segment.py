# -*- coding: utf-8 -*-
import sys
from prob_emit import P as emit_prob
from prob_start import P as start_prob
from prob_trans import P as trans_prob

# reverse_trans = {}
# for start, v in trans_prob.items():
#   for dest, prob in v.items():
#       reverse_trans.setdefault(dest, {})
#       reverse_trans[dest][start] = prob


def do_hmm(char_list):
    last_idx = len(char_list) - 1
    status_list = []

    all_states = ["B", "M", "E", "S"]
    for idx, char in enumerate(char_list):
        cur_prob = {}
        status_list.append(cur_prob)
        for each in all_states:
            cur_prob[each] = [emit_prob[each].get(char, -20), ""]

        if idx == last_idx:
            del cur_prob["B"]
            del cur_prob["M"]

        if idx == 0:
            for each in cur_prob.keys():
                cur_prob[each][0] += start_prob[each]
        else:
            last_prob = status_list[idx - 1]
            for cur_s, cur_val in cur_prob.items():
                max_trans = -10000
                max_last_stats = ""
                for last_s, last_val in last_prob.items():
                    if last_s not in trans_prob or cur_s not in trans_prob[last_s]:
                        continue
                    if last_val[0] + trans_prob[last_s][cur_s] > max_trans:
                        max_trans = last_val[0] + trans_prob[last_s][cur_s]
                        max_last_stats = last_s
                cur_prob[cur_s][0] += max_trans
                cur_prob[cur_s][1] = max_last_stats

    last_stats = status_list[last_idx]
    max_last = "S" if last_stats["S"][0] > last_stats["E"][0] else "E"
    final_tag = ["" for x in range(len(char_list))]
    final_tag[-1] = max_last
    for idx in range(last_idx, 0, -1):
        cur_tag = status_list[idx][max_last][1]
        final_tag[idx - 1] = cur_tag
        max_last = cur_tag
    print final_tag
    res = []
    cur_string_list = []
    for char, tag in zip(char_list, final_tag):
        cur_string_list.append(char)

        if tag in ("S", "E"):
            res.append(''.join(cur_string_list))
            cur_string_list = []

    return " ".join(res)


def segment(sentence):
    return do_hmm(sentence.decode('utf8'))


if __name__ == "__main__":
    print segment("监督学习的模型分为生成模型与判别模型")
