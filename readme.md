# Distributed RAG Architecture

## Overview

This project implements a distributed Retrieval-Augmented Generation (RAG) architecture designed to integrate multiple data sources, specialized agents, and large language models (LLMs) for enhanced information retrieval and decision-making processes.

### Key Features

- **Multi-Database Integration**: Supports Vector DB, RDBMS, and NoSQL databases.
- **Specialized Agents**: Includes Code, Cyber Security, Project Management, and Cyber Research agents.
- **Enhanced Consensus Mechanism**: Implements a sophisticated vote-based decision-making algorithm.
- **LLM Integration**: Incorporates multiple LLMs for various reasoning tasks.
- **Robust Error Handling**: Comprehensive error management for database operations, API calls, and general exceptions.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/distributed-rag-architecture.git
   cd distributed-rag-architecture
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your `config.json` file with the necessary configurations for databases, APIs, and LLMs.

## Usage

1. Ensure your `config.json` file is properly set up with your database connections, API endpoints, and LLM configurations.

2. Run the main script:
   ```
   python main.py
   ```

3. The script will demonstrate the workflow of the distributed RAG architecture, including:
   - Task assignment by the Project Manager agent
   - Code generation by the Code agent
   - Security evaluation by the Cyber Security agent
   - Threat research by the Cyber Research agent
   - LLM-based reasoning
   - Consensus-based decision making
   - Database queries and API calls

## Configuration

The `config.json` file should include the following sections:

- `databases`: Configuration for Vector DB, RDBMS, and NoSQL databases.
- `apis`: Endpoints for external API calls.
- `llm`: Settings for different LLMs used by various agents.

Example structure:

```json
{
  "databases": {
    "vector_db": { "host": "vector_db_host", "port": 5000 },
    "rdbms": { "host": "rdbms_host", "port": 3306 },
    "nosql_db": { "host": "nosql_host", "port": 27017 }
  },
  "apis": {
    "security_updates": "https://api.example.com/security-updates"
  },
  "llm": {
    "security": { "name": "SecurityLLM", "endpoint": "https://llm.example.com/security" },
    "code": { "name": "CodeLLM", "endpoint": "https://llm.example.com/code" }
  }
}
```

## Extending the Project

To extend this project:

1. Add new agent types by creating new classes that inherit from the `Agent` base class.
2. Implement new database connectors by extending the `Database` base class.
3. Enhance the consensus mechanism by modifying the `EnhancedConsensusMechanism` class.
4. Add new LLM integrations in the `LLMReasoning` class.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project was inspired by the need for more sophisticated RAG architectures in AI applications.
- Special thanks to the open-source community for providing valuable tools and libraries that made this project possible.
