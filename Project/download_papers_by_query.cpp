#include <mpi.h>
#include <omp.h>
#include <curl/curl.h>
#include "tinyxml2.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>

using namespace tinyxml2;

// Helper function to write data into a std::string
size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}


// Download file to disk, validate afterwards
bool download_file_with_curl(CURL* curl, const std::string& url, const std::string& filepath) {
    if (!curl) {
        std::cerr << "CURL handle is null!" << std::endl;
        return false;
    }

    FILE* fp = fopen(filepath.c_str(), "wb");
    if (!fp) {
        std::cerr << "Failed to open " << filepath << ": " << strerror(errno) << std::endl;
        return false;
    }

    curl_easy_reset(curl);
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, NULL);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 5L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "Mozilla/5.0 (Windows NT 10.0; Win64; x64)");
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
    curl_easy_setopt(curl, CURLOPT_STDERR, stderr);
    curl_easy_setopt(curl, CURLOPT_SSLVERSION, CURL_SSLVERSION_TLSv1_2);

    std::cout << "Attempting to download: " << url << std::endl;
    CURLcode res = curl_easy_perform(curl);
    fclose(fp);

    if (res != CURLE_OK) {
        std::cerr << "CURL failed (" << res << "): " << curl_easy_strerror(res) << std::endl;
        remove(filepath.c_str());
        return false;
    }

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    std::cout << "HTTP response code: " << http_code << std::endl;

    return true;
}

// Fetch XML string from arXiv API
std::string fetch_xml(const std::string& url) {
    CURL* curl = curl_easy_init();
    std::string readBuffer;

    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
        curl_easy_setopt(curl, CURLOPT_USERAGENT, "arXiv-Downloader/1.0");

        CURLcode res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            std::cerr << "curl_easy_perform() failed: " << curl_easy_strerror(res) << std::endl;
        }
        curl_easy_cleanup(curl);
    }
    return readBuffer;
}

// Sanitize a file name
std::string sanitize_filename(std::string filename) {
    std::replace_if(filename.begin(), filename.end(), 
        [](char c) { return !isalnum(c) && c != '_' && c != '-' && c != '.'; }, '_');
    return filename;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    curl_global_init(CURL_GLOBAL_DEFAULT);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <query> <number of papers>" << std::endl;
        }
        MPI_Finalize();
        curl_global_cleanup();
        return 1;
    }

    std::string query = argv[1];
    int max_results = std::stoi(argv[2]);
    std::string api_url = "http://export.arxiv.org/api/query?search_query=" + query + 
                         "&max_results=" + std::to_string(max_results) + "&sortBy=submittedDate";

    std::string xml_data;
    if (rank == 0) {
        xml_data = fetch_xml(api_url);
        if (xml_data.empty()) {
            std::cerr << "Failed to fetch data from arXiv API" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    int xml_length;
    if (rank == 0) {
        xml_length = static_cast<int>(xml_data.size());
    }
    MPI_Bcast(&xml_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<char> xml_buffer(xml_length + 1);
    if (rank == 0) {
        std::copy(xml_data.begin(), xml_data.end(), xml_buffer.begin());
        xml_buffer[xml_length] = '\0';
    }
    MPI_Bcast(xml_buffer.data(), xml_length + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    xml_data = std::string(xml_buffer.data());

    XMLDocument doc;
    XMLError parse_result = doc.Parse(xml_data.c_str());
    if (parse_result != XML_SUCCESS) {
        if (rank == 0) {
            std::cerr << "Failed to parse XML feed. Error: " << doc.ErrorStr() << std::endl;
        }
        MPI_Finalize();
        curl_global_cleanup();
        return 1;
    }

    XMLElement* feed = doc.FirstChildElement("feed");
    if (!feed) {
        if (rank == 0) {
            std::cerr << "No feed element found in XML" << std::endl;
        }
        MPI_Finalize();
        curl_global_cleanup();
        return 1;
    }

    std::vector<XMLElement*> entries;
    for (XMLElement* entry = feed->FirstChildElement("entry"); entry != nullptr; entry = entry->NextSiblingElement("entry")) {
        entries.push_back(entry);
    }

    int total = static_cast<int>(entries.size());
    if (total == 0) {
        if (rank == 0) {
            std::cerr << "No entries found in the feed" << std::endl;
        }
        MPI_Finalize();
        curl_global_cleanup();
        return 1;
    }

    int chunk = (total + size - 1) / size;
    int start = rank * chunk;
    int end = std::min(start + chunk, total);

    if (rank == 0) {
        if (mkdir("pdfs", 0777) != 0 && errno != EEXIST) {
            std::cerr << "Failed to create pdfs directory: " << strerror(errno) << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    #pragma omp parallel
    {
        CURL* thread_curl = curl_easy_init();

        #pragma omp for schedule(dynamic)
        for (int i = start; i < end; ++i) {
            XMLElement* entry = entries[i];

            XMLElement* titleElement = entry->FirstChildElement("title");
            if (!titleElement || !titleElement->GetText()) {
                #pragma omp critical
                std::cerr << "[Rank " << rank << "] Entry has no title" << std::endl;
                continue;
            }

            XMLElement* idElement = entry->FirstChildElement("id");
            if (!idElement || !idElement->GetText()) {
                #pragma omp critical
                std::cerr << "[Rank " << rank << "] Entry has no ID" << std::endl;
                continue;
            }

            std::string safe_title = sanitize_filename(titleElement->GetText());
            std::string arxiv_id_url = idElement->GetText();
            std::string arxiv_id = arxiv_id_url.substr(arxiv_id_url.find_last_of('/') + 1);

            std::string pdf_url = "http://arxiv.org/pdf/" + arxiv_id + ".pdf";

            std::string filepath = "pdfs/" + safe_title + ".pdf";
            std::cout << "[Rank " << rank << ", Thread " << omp_get_thread_num() << "] Downloading: " << safe_title << std::endl;

            bool success = download_file_with_curl(thread_curl, pdf_url, filepath);
            if (!success) {
                #pragma omp critical
                std::cerr << "[Rank " << rank << "] Failed to download: " << pdf_url << std::endl;
            }
        }

        curl_easy_cleanup(thread_curl);
    }

    MPI_Finalize();
    curl_global_cleanup();
    return 0;
}
