#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_set>
#include <filesystem>
#include <algorithm>
#include <mutex>
#include "json.hpp"
#include <omp.h> // For OpenMP

namespace fs = std::filesystem;
using json = nlohmann::json;

// Utility: trim whitespace from start and end
std::string trim(const std::string& str) {
    const auto strBegin = str.find_first_not_of(" \n\r\t");
    if (strBegin == std::string::npos)
        return "";
    const auto strEnd = str.find_last_not_of(" \n\r\t");
    return str.substr(strBegin, strEnd - strBegin + 1);
}

void to_json_cpp_parallel(const std::vector<std::string>& output_list, const std::vector<std::string>& docs) {
    std::vector<json> q_a_list;
    std::mutex q_a_list_mutex;

    // Parse outputs in parallel
    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < output_list.size(); ++i) {
        const auto& output = output_list[i];
        size_t pos = 0;
        while ((pos = output.find('{', pos)) != std::string::npos) {
            size_t end_pos = output.find('}', pos);
            if (end_pos != std::string::npos) {
                std::string candidate = output.substr(pos, end_pos - pos + 1);
                candidate = trim(candidate);
                try {
                    auto parsed = json::parse(candidate);
                    if (!candidate.empty() && candidate.back() == '"') {
                        std::lock_guard<std::mutex> lock(q_a_list_mutex);
                        q_a_list.push_back(parsed);
                    }
                } catch (...) {
                    // Ignore invalid JSON
                }
                pos = end_pos + 1;
            } else {
                break;
            }
        }
    }

    // Filter valid entries in parallel
    std::vector<json> filtered;
    std::mutex filtered_mutex;

    #pragma omp parallel for schedule(dynamic)
    for (size_t i = 0; i < q_a_list.size(); ++i) {
        const auto& elem = q_a_list[i];
        bool present = false;

        if (elem.contains("keywords") && !elem["keywords"].is_null()) {
            try {
                for (const auto& doc : docs) {
                    for (const auto& keyword : elem["keywords"]) {
                        std::string keyword_lower = keyword.get<std::string>();
                        std::transform(keyword_lower.begin(), keyword_lower.end(), keyword_lower.begin(), ::tolower);

                        std::string doc_lower = doc;
                        std::transform(doc_lower.begin(), doc_lower.end(), doc_lower.begin(), ::tolower);
                        doc_lower.erase(std::remove(doc_lower.begin(), doc_lower.end(), '\n'), doc_lower.end());

                        if (doc_lower.find(keyword_lower) != std::string::npos) {
                            present = true;
                            break;
                        }
                    }
                    if (present) break;
                }
            } catch (...) {
                // Ignore problematic keywords
            }
        }

        bool valid_fields = elem.contains("instruction") && elem.contains("keywords")
                         && elem.contains("output") && elem.contains("source")
                         && elem.contains("context");

        bool non_empty = true;
        for (const auto& field : {"instruction", "keywords", "output", "source", "context"}) {
            if (elem[field].is_null() || (elem[field].is_string() && elem[field].get<std::string>().empty())) {
                non_empty = false;
                break;
            }
        }

        bool has_question_mark = elem.contains("instruction") && elem["instruction"].get<std::string>().find('?') != std::string::npos;
        bool valid_output = elem.contains("output") &&
                            (elem["output"].get<std::string>().find("no information") == std::string::npos &&
                             elem["output"].get<std::string>().find("not covered") == std::string::npos);

        if (valid_fields && non_empty && has_question_mark && valid_output && present) {
            std::lock_guard<std::mutex> lock(filtered_mutex);
            filtered.push_back(elem);
        }
    }

    // Remove duplicates inside the batch
    std::vector<json> unique_filtered;
    std::unordered_set<std::string> seen_batch;
    for (const auto& elem : filtered) {
        std::string dump = elem.dump();
        if (seen_batch.find(dump) == seen_batch.end()) {
            unique_filtered.push_back(elem);
            seen_batch.insert(dump);
        }
    }

    // Prepare directory
    fs::create_directories("jsons");

    // Read existing JSON
    json existing = json::array();
    if (fs::exists("jsons/arxiv_version2.json")) {
        std::ifstream in_file("jsons/arxiv_version2.json");
        try {
            in_file >> existing;
        } catch (...) {
            std::cerr << "Existing file corrupt or empty, starting fresh.\n";
        }
        in_file.close();
    }

    // Build a set of existing entries
    std::unordered_set<std::string> seen_existing;
    for (const auto& e : existing) {
        seen_existing.insert(e.dump());
    }

    // Append new entries without duplication
    for (const auto& elem : unique_filtered) {
        std::string elem_str = elem.dump();
        if (seen_existing.find(elem_str) == seen_existing.end()) {
            existing.push_back(elem);
            seen_existing.insert(elem_str);
        }
    }

    // Save to file
    std::ofstream out_file("jsons/arxiv_version2.json");
    out_file << existing.dump(2); // Pretty print
    out_file.close();
}


int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <output_list.json> <docs.json>\n";
        return 1;
    }

    std::ifstream f1(argv[1]);
    std::ifstream f2(argv[2]);
    if (!f1.is_open() || !f2.is_open()) {
        std::cerr << "Error opening input files.\n";
        return 1;
    }

    json output_list_json, docs_json;
    f1 >> output_list_json;
    f2 >> docs_json;

    std::vector<std::string> output_list = output_list_json.get<std::vector<std::string>>();
    std::vector<std::string> docs = docs_json.get<std::vector<std::string>>();

    to_json_cpp_parallel(output_list, docs);

    return 0;
}
