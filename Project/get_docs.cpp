#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_set>
#include <filesystem>
#include <algorithm>
#include <memory>
#include <omp.h>

// pdf reading libriries
#include <poppler-document.h>
#include <poppler-page.h>

namespace fs = std::filesystem;

// Structure to hold parsed PDF entry
struct ParsedEntry {
    std::string filename;
    std::string content;
};

// Clean resultant string from parsing the pdfs: remove extra spaces, replace \n\t\r with spaces
// this is done to make the text more readable and to avoid issues with JSON formatting
std::string clean_text(const std::string& input) {
    std::string output;
    bool lastWasSpace = false;

    for (char c : input) {
        if (c == '\n' || c == '\r' || c == '\t') {
            c = ' ';
        }

        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!lastWasSpace) {
                output += ' ';
                lastWasSpace = true;
            }
        } else {
            output += c;
            lastWasSpace = false;
        }
    }

    if (!output.empty() && output.back() == ' ') {
        output.pop_back();
    }

    return output;
}

// Escape a string to make it JSON-safe
std::string escape_json_string(const std::string& input) {
    std::ostringstream oss;
    for (char c : input) {
        switch (c) {
            case '\"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buffer[7];
                    std::snprintf(buffer, sizeof(buffer), "\\u%04X", c & 0xFF);
                    oss << buffer;
                } else {
                    oss << c;
                }
        }
    }
    return oss.str();
}

int main() {
    std::string pdf_folder = "pdfs/";
    std::vector<ParsedEntry> parsed;
    std::unordered_set<std::string> hash_lines;

    // Locks for thread safety
    // necessary for vector and hash set
    omp_lock_t hash_lock;
    omp_lock_t parsed_lock;
    omp_init_lock(&hash_lock);
    omp_init_lock(&parsed_lock);

    // Gather all PDF files
    std::vector<fs::directory_entry> pdf_files;
    for (const auto& entry : fs::directory_iterator(pdf_folder)) {
        if (entry.path().extension() == ".pdf") {
            pdf_files.push_back(entry);
        }
    }

    int num_pdfs = pdf_files.size();

    // Parallel parsing
    #pragma omp parallel
    {
        std::vector<ParsedEntry> local_parsed; // Thread-local buffer

        #pragma omp for schedule(dynamic)
        for (int idx = 0; idx < num_pdfs; ++idx) {
            const auto& entry = pdf_files[idx];
            std::string filename = entry.path().filename().string();
            std::string filepath = entry.path().string();

            bool is_already_present = false;

            // Load PDF
            std::unique_ptr<poppler::document> doc(poppler::document::load_from_file(filepath));
            if (!doc) {
                continue;
            }

            int num_pages = doc->pages();

            for (int i = 0; i < num_pages; ++i) {
                std::unique_ptr<poppler::page> page(doc->create_page(i));
                if (!page) {
                    continue;
                }

                auto byte_vec = page->text().to_utf8();
                std::string content(byte_vec.begin(), byte_vec.end());
                content = clean_text(content); // <<< clean the text here

                if (i == 0) { // first page special logic
                    std::string cleaned = clean_text(content);
                    size_t abstract_pos = cleaned.find("abstract");
                    if (abstract_pos != std::string::npos) {
                        cleaned = cleaned.substr(0, abstract_pos);
                    }

                    bool already_present = false;
                    omp_set_lock(&hash_lock);
                    if (hash_lines.find(cleaned) != hash_lines.end()) {
                        already_present = true;
                    } else {
                        hash_lines.insert(cleaned);
                    }
                    omp_unset_lock(&hash_lock);

                    if (already_present) {
                        is_already_present = true;
                        break; // skip this PDF
                    } else {
                        ParsedEntry parsed_entry;
                        parsed_entry.filename = filename;
                        parsed_entry.content = cleaned;
                        local_parsed.push_back(parsed_entry);
                    }
                } else {
                    if (!is_already_present) {
                        ParsedEntry parsed_entry;
                        parsed_entry.filename = filename;
                        parsed_entry.content = content;
                        local_parsed.push_back(parsed_entry);
                    }
                }
            }
        }

        // Merge local parsed into global parsed
        omp_set_lock(&parsed_lock);
        parsed.insert(parsed.end(), local_parsed.begin(), local_parsed.end());
        omp_unset_lock(&parsed_lock);
    }

    // Destroy locks
    omp_destroy_lock(&hash_lock);
    omp_destroy_lock(&parsed_lock);

    // Print final parsed JSON to stdout so that the main python script can consume it
    std::cout << "[";
    for (size_t i = 0; i < parsed.size(); ++i) {
        std::cout << "{\"" 
                  << escape_json_string(parsed[i].filename)
                  << "\":\""
                  << escape_json_string(parsed[i].content)
                  << "\"}";
        if (i != parsed.size() - 1) {
            std::cout << ",";
        }
    }
    std::cout << "]";

    return 0;
}
