#!/usr/bin/env python3
"""Submission validator for Amazon ML Challenge 2026: Business Entity Resolution.
Checks matching_results.tsv and candidate_pairs.tsv against official rules.
"""
import argparse
import os
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Validate submission files.")
    parser.add_argument("--matching", required=True, help="Path to matching_results.tsv")
    parser.add_argument("--candidate", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--test-dir", required=True, help="Path to test directory (containing test_source1.tsv, etc.)")
    parser.add_argument("--check-ids", action="store_true", default=True, help="Validate entity IDs exist in test set")
    return parser.parse_args()

def load_ids(tsv_path):
    ids = set()
    with open(tsv_path, "r", encoding="utf-8") as f:
        header = f.readline()
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            ids.add(parts[0])
    return ids

def validate(matching_path, candidate_path, test_dir, check_ids=True):
    issues = []

    # 1. Check file existence
    if not os.path.exists(matching_path):
        issues.append(f"Matching file not found: {matching_path}")
    if not os.path.exists(candidate_path):
        issues.append(f"Candidate file not found: {candidate_path}")
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    for p in (s1_path, s2_path, s3_path):
        if not os.path.exists(p):
            issues.append(f"Test file not found: {p}")

    if issues:
        return issues

    print("Loading test source IDs...")
    valid_s1 = load_ids(s1_path)
    valid_s2_s3 = load_ids(s2_path) | load_ids(s3_path) if check_ids else set()

    # 2. Check matching_results.tsv
    print("Checking matching_results.tsv...")
    s1_in_matching = []
    matching_pairs = {}
    with open(matching_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if header != ["source1_entity_id", "matched_entity_ids"]:
            issues.append(f"Invalid header in {matching_path}: expected ['source1_entity_id', 'matched_entity_ids'], got {header}")
        for lineno, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 1:
                s1_id = parts[0]
                matched_str = ""
            elif len(parts) == 2:
                s1_id, matched_str = parts
            else:
                issues.append(f"Line {lineno} in {matching_path} has {len(parts)} columns (expected 2)")
                continue

            s1_in_matching.append(s1_id)
            if matched_str:
                m_ids = matched_str.split(",")
                if len(m_ids) != len(set(m_ids)):
                    issues.append(f"Duplicate IDs in matched_entity_ids for {s1_id} at line {lineno}")
                if check_ids:
                    for mid in m_ids:
                        if not (mid.startswith("S2-") or mid.startswith("S3-")):
                            issues.append(f"Invalid ID prefix for {mid} at line {lineno} (must start with S2- or S3-)")
                            break
                        if mid not in valid_s2_s3:
                            issues.append(f"Entity ID {mid} at line {lineno} does not exist in test set")
                            break
                matching_pairs[s1_id] = set(m_ids)
            else:
                matching_pairs[s1_id] = set()

    # Check S1 coverage
    s1_set = set(s1_in_matching)
    if len(s1_in_matching) != len(s1_set):
        issues.append(f"Duplicate source1_entity_id rows in {matching_path}")
    if s1_set != valid_s1:
        missing = valid_s1 - s1_set
        extra = s1_set - valid_s1
        if missing:
            issues.append(f"{len(missing)} Source 1 entities missing from {matching_path}")
        if extra:
            issues.append(f"{len(extra)} unknown Source 1 entities in {matching_path}")

    # 3. Check candidate_pairs.tsv
    print("Checking candidate_pairs.tsv...")
    s1_in_candidate = []
    candidate_pairs = {}
    with open(candidate_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if header != ["source1_entity_id", "candidate_entity_ids"]:
            issues.append(f"Invalid header in {candidate_path}: expected ['source1_entity_id', 'candidate_entity_ids'], got {header}")
        for lineno, line in enumerate(f, start=2):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 1:
                s1_id = parts[0]
                cand_str = ""
            elif len(parts) == 2:
                s1_id, cand_str = parts
            else:
                issues.append(f"Line {lineno} in {candidate_path} has {len(parts)} columns (expected 2)")
                continue

            s1_in_candidate.append(s1_id)
            if cand_str:
                c_ids = cand_str.split(",")
                if len(c_ids) != len(set(c_ids)):
                    issues.append(f"Duplicate IDs in candidate_entity_ids for {s1_id} at line {lineno}")
                candidate_pairs[s1_id] = set(c_ids)
            else:
                candidate_pairs[s1_id] = set()

    cand_set = set(s1_in_candidate)
    if len(s1_in_candidate) != len(cand_set):
        issues.append(f"Duplicate source1_entity_id rows in {candidate_path}")
    if cand_set != valid_s1:
        missing = valid_s1 - cand_set
        if missing:
            issues.append(f"{len(missing)} Source 1 entities missing from {candidate_path}")

    # 4. Check that matched_entity_ids is a subset of candidate_entity_ids
    print("Checking subset consistency...")
    subset_violations = 0
    for s1_id, m_set in matching_pairs.items():
        c_set = candidate_pairs.get(s1_id, set())
        diff = m_set - c_set
        if diff:
            subset_violations += 1
            if subset_violations <= 5:
                issues.append(f"Matched ID(s) {diff} for {s1_id} not in candidate set")
    if subset_violations > 5:
        issues.append(f"... and {subset_violations - 5} more subset violations")

    return issues

def main():
    args = parse_args()
    issues = validate(args.matching, args.candidate, args.test_dir, args.check_ids)
    if not issues:
        print("\n==========================================")
        print("PASS: Both files are 100% valid and safe to submit!")
        print("==========================================")
        sys.exit(0)
    else:
        print("\n==========================================")
        print(f"FAILED with {len(issues)} issue(s):")
        for i, issue in enumerate(issues[:30], 1):
            print(f"{i}. {issue}")
        if len(issues) > 30:
            print(f"... and {len(issues) - 30} more issues")
        print("==========================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
