# 🤖 Agent Framework: Simple Yet Powerful Workflow Orchestration

Build complex workflows as easily as writing Python functions! Perfect for AI/ML pipelines, data processing, automation, and more.

## Why Use Agent Framework? 

### 🎯 Perfect For:
- **AI/ML Engineers**: Orchestrate LLM pipelines and ML workflows
- **Data Engineers**: Build ETL and data processing flows
- **Backend Developers**: Create robust service workflows
- **Automation Engineers**: Design complex automation systems
- **Researchers**: Coordinate experiments and analysis

### 🌟 Key Benefits:
- **Write Less Code**: Turn complex workflows into simple state functions
- **No More Spaghetti Code**: Clean, organized workflow management
- **Type Safety**: Catch errors before they happen
- **Automatic Resource Management**: No manual resource tracking
- **Built for Scale**: From simple scripts to production systems

## 🎓 Learn by Example

### Example 1: Text Processing Pipeline

Perfect for: Processing documents, articles, or any text data

```python
from agent import Agent, Context, StateResult

# Create agent
agent = Agent("text_processor")

async def clean_text(context: Context) -> StateResult:
    text = "Hello   World!  "
    
    # Store processed text
    cleaned = text.strip().replace("  ", " ")
    context.set_state("cleaned_text", cleaned)
    
    return "analyze_text"

async def analyze_text(context: Context) -> StateResult:
    # Get text from previous state
    text = context.get_state("cleaned_text")
    
    # Simple analysis
    stats = {
        "length": len(text),
        "words": len(text.split()),
        "has_punctuation": "!" in text
    }
    context.set_state("stats", stats)
    
    return "format_results"

async def format_results(context: Context) -> StateResult:
    stats = context.get_state("stats")
    text = context.get_state("cleaned_text")
    
    # Format final results
    result = {
        "text": text,
        "analysis": stats,
        "timestamp": time.time()
    }
    context.set_state("final_result", result)
    
    return None  # End workflow

# Add states
agent.add_state("clean_text", clean_text)
agent.add_state("analyze_text", analyze_text)
agent.add_state("format_results", format_results)

# Run it!
await agent.run()
```

### Example 2: LLM Processing Pipeline

Perfect for: Working with AI models and processing results

```python
from dataclasses import dataclass
from typing import List

@dataclass
class LLMResponse(TypedContextData):
    text: str
    confidence: float
    tokens_used: int
    model: str

async def generate_text(context: Context) -> StateResult:
    # Simulate LLM call
    response = LLMResponse(
        text="AI generated text here...",
        confidence=0.95,
        tokens_used=150,
        model="gpt-4"
    )
    
    context.set_typed("llm_response", response)
    return ["check_quality", "extract_entities"]

async def check_quality(context: Context) -> StateResult:
    response = context.get_typed("llm_response", LLMResponse)
    
    if response.confidence < 0.9:
        return "regenerate_text"
    
    return "await_processing"

async def extract_entities(context: Context) -> StateResult:
    response = context.get_typed("llm_response", LLMResponse)
    # Process entities...
    return "await_processing"

# Add states and run
agent = Agent("llm_processor")
agent.add_state("generate_text", generate_text)
agent.add_state("check_quality", check_quality)
agent.add_state("extract_entities", extract_entities)
await agent.run()
```

### Example 3: Data Processing Pipeline

Perfect for: ETL jobs, data transformations, and analysis

```python
@dataclass
class DataBatch(TypedContextData):
    data: List[dict]
    batch_id: str
    timestamp: float

async def load_data(context: Context) -> StateResult:
    # Simulate data load
    batch = DataBatch(
        data=[{"id": 1, "value": "test"}],
        batch_id="batch_001",
        timestamp=time.time()
    )
    
    context.set_typed("batch", batch)
    return ["validate_data", "transform_data"]

async def validate_data(context: Context) -> StateResult:
    batch = context.get_typed("batch", DataBatch)
    
    # Validate each record
    valid = all(
        isinstance(record.get('id'), int)
        for record in batch.data
    )
    
    if not valid:
        return "handle_error"
        
    context.set_state("validation_passed", True)
    return "await_processing"

# Add states with resource management
agent = Agent("data_processor")
agent.add_state(
    "load_data", 
    load_data,
    resources=ResourceRequirements(
        cpu_units=1.0,
        memory_mb=512
    )
)
```

### Example 4: ML Training Pipeline

Perfect for: Managing model training workflows

```python
@dataclass
class TrainingConfig(TypedContextData):
    model_name: str
    batch_size: int
    epochs: int
    learning_rate: float

@dataclass
class TrainingMetrics(TypedContextData):
    loss: float
    accuracy: float
    epoch: int

async def prepare_training(context: Context) -> StateResult:
    config = TrainingConfig(
        model_name="my_model",
        batch_size=32,
        epochs=10,
        learning_rate=0.001
    )
    
    context.set_typed("config", config)
    return "train_model"

async def train_model(context: Context) -> StateResult:
    config = context.get_typed("config", TrainingConfig)
    
    for epoch in range(config.epochs):
        # Simulate training
        metrics = TrainingMetrics(
            loss=0.1,
            accuracy=0.95,
            epoch=epoch
        )
        context.set_typed(f"metrics_epoch_{epoch}", metrics)
        
        # Early stopping
        if metrics.accuracy > 0.98:
            return "evaluate_model"
    
    return "evaluate_model"
```

## 🎯 More Real-World Use Cases

### Content Management System
```python
async def process_article(context: Context) -> StateResult:
    # Process new article
    article = context.get_typed("article", Article)
    
    # Run parallel processes
    return [
        "spell_check",
        "generate_summary",
        "extract_keywords",
        "create_images"
    ]

async def spell_check(context: Context) -> StateResult:
    # Check spelling
    return "await_completion"

async def generate_summary(context: Context) -> StateResult:
    # Generate AI summary
    return "await_completion"

async def create_images(context: Context) -> StateResult:
    # Generate images with DALL-E
    return "await_completion"

async def await_completion(context: Context) -> StateResult:
    # Combine all results
    return "publish"
```

### Customer Service Automation
```python
async def handle_ticket(context: Context) -> StateResult:
    ticket = context.get_typed("ticket", ServiceTicket)
    
    # Analyze ticket priority
    if ticket.priority == "high":
        return "immediate_response"
    elif "bug" in ticket.tags:
        return "technical_review"
    else:
        return "standard_response"

async def immediate_response(context: Context) -> StateResult:
    # Generate quick response
    response = await generate_priority_response()
    context.set_state("response", response)
    
    return "human_review"
```

### Data Analysis Pipeline
```python
async def analyze_dataset(context: Context) -> StateResult:
    dataset = context.get_typed("dataset", Dataset)
    
    # Split processing based on size
    if len(dataset.data) > 10000:
        chunks = split_data(dataset.data, 5)
        for i, chunk in enumerate(chunks):
            context.set_state(f"chunk_{i}", chunk)
        return ["process_chunk_0", "process_chunk_1", "process_chunk_2"]
    else:
        return "simple_process"

async def process_chunk(context: Context) -> StateResult:
    # Process data chunk
    chunk_id = context.get_state("chunk_id")
    data = context.get_state(f"chunk_{chunk_id}")
    
    results = analyze_data(data)
    context.set_state(f"results_{chunk_id}", results)
    
    return "merge_results"
```

## 🚀 Getting Started

1. Install the framework:
```bash
pip install agent-framework
```

2. Create your first workflow:
```python
from agent import Agent, Context, StateResult

# Create agent
agent = Agent("my_first_workflow")

# Define states
async def first_step(context: Context) -> StateResult:
    context.set_state("message", "Hello World!")
    return "second_step"

async def second_step(context: Context) -> StateResult:
    msg = context.get_state("message")
    print(f"Got message: {msg}")
    return None

# Add states and run
agent.add_state("first_step", first_step)
agent.add_state("second_step", second_step)
await agent.run()
```

## 💡 Best Practices

1. **Keep States Focused**
   - Each state should do one thing well
   - Use multiple states instead of complex functions

2. **Use Type Safety**
   - Define data classes for structured data
   - Let the framework catch type errors early

3. **Handle Errors Gracefully**
   - Use retry policies for unreliable operations
   - Add error states for proper fallback

4. **Monitor Resources**
   - Set appropriate resource requirements
   - Use parallel processing wisely

## 🌟 Join Our Community!

- Share your workflows
- Get help from experts
- Contribute to the framework
- Stay updated on new features

## License

MIT License - Free for personal and commercial use!

## ✨ All Core Features

### 1. Smart Context System
The context system lets you share data between states with type safety and automatic tracking.

```python
from fluffy_agent import Agent, Context, StateResult

# Untyped - Simple and flexible
async def store_data(context: Context) -> StateResult:
    # Store any data type
    context.set_state("user_name", "Alice")
    context.set_state("scores", [95, 87, 92])
    return "next_state"

# Typed - Safe and powerful
from dataclasses import dataclass
from typing import List

@dataclass
class UserProfile(TypedContextData):
    name: str
    age: int
    scores: List[float]

async def handle_user(context: Context) -> StateResult:
    # Type-safe storage
    profile = UserProfile(name="Alice", age=28, scores=[95.0, 87.5])
    context.set_typed("user", profile)
    
    # Type-safe updates
    context.update_typed("user", age=29)
    return "next_state"
```

### 2. Flow Control
Easily control your workflow with sequential and parallel execution.

```python
# Sequential Flow
async def step1(context: Context) -> StateResult:
    context.set_state("data", "processed")
    return "step2"  # Go to step2

async def step2(context: Context) -> StateResult:
    return "step3"  # Go to step3

# Parallel Flow
async def start_parallel(context: Context) -> StateResult:
    return ["process_images", "process_text"]  # Run both in parallel

async def merge_results(context: Context) -> StateResult:
    # Automatically waits for both processes
    return None
```

### 3. Resource Management
Automatically manage and optimize resource usage.

```python
from fluffy_agent import ResourceRequirements, Priority

# Define resources needed
requirements = ResourceRequirements(
    cpu_units=2.0,
    memory_mb=1024,
    gpu_units=1.0,
    priority=Priority.HIGH
)

# Add state with resources
agent.add_state(
    "train_model",
    train_model,
    resources=requirements
)
```

### 4. Dependency Management
Define complex dependencies between states.

```python
from fluffy_agent import DependencyType, DependencyLifecycle

agent.add_state(
    "final_step",
    final_step,
    dependencies={
        # Must complete first
        "prepare_data": DependencyType.REQUIRED,
        
        # Optional dependency
        "validate_data": DependencyType.OPTIONAL,
        
        # Run in parallel
        "background_task": DependencyType.PARALLEL
    }
)
```

### 5. Error Handling & Retries
Built-in error handling and retry mechanisms.

```python
from fluffy_agent import RetryPolicy

# Configure retries
retry_policy = RetryPolicy(
    max_retries=3,
    initial_delay=1.0,
    exponential_base=2.0,
    jitter=True
)

# Add state with retry policy
agent.add_state(
    "api_call",
    make_api_call,
    retry_policy=retry_policy
)
```

### 6. Human-in-the-Loop Integration
Easily integrate human feedback into your workflows.

```python
async def review_content(context: Context) -> StateResult:
    content = context.get_state("content")
    
    feedback = await context.human_in_the_loop(
        prompt="Review this content:",
        timeout=300,
        validator=lambda x: len(x) > 10
    )
    
    if "approve" in feedback.lower():
        return "publish"
    else:
        return "revise"
```

### 7. Agent Execution Control
Control how your workflow runs with flexible execution options.

```python
# Basic execution
await agent.run()

# With timeout
await agent.run(timeout=300)  # 5 minute timeout

# With cleanup handling
await agent.run(cleanup_timeout=60)  # 1 minute cleanup

# Configure maximum concurrent states
agent = Agent(
    "my_workflow",
    max_concurrent=5  # Max 5 states running at once
)

# Pause and resume
checkpoint = await agent.pause()  # Pauses execution and returns checkpoint
await agent.resume()  # Resumes from pause

# Cancel execution
await agent.cancel_all()  # Cancels all running states
```

### 8. Checkpointing & Recovery
Save and restore workflow state at any point.

```python
# Create checkpoint
checkpoint = agent.create_checkpoint()

# Save checkpoint to file
import json
with open("workflow_state.json", "w") as f:
    json.dump(checkpoint.to_dict(), f)

# Create new agent and restore
new_agent = Agent("restored_workflow")
await new_agent.restore_from_checkpoint(checkpoint)

# Automatic periodic checkpointing
agent = Agent(
    "my_workflow",
    checkpoint_interval=300  # Checkpoint every 5 minutes
)

# Get specific checkpoint data
checkpoint_data = {
    "timestamp": checkpoint.timestamp,
    "agent_name": checkpoint.agent_name,
    "agent_status": checkpoint.agent_status,
    "completed_states": list(checkpoint.completed_states),
    "shared_state": checkpoint.shared_state
}
```

### 9. State Execution Lifecycle
Monitor and control state execution status.

```python
from fluffy_agent import AgentStatus, StateStatus

# Check agent status
status = agent.status  # IDLE, RUNNING, PAUSED, COMPLETED, FAILED

# Check state status
state_status = agent.state_metadata["my_state"].status
# PENDING, READY, RUNNING, COMPLETED, FAILED, BLOCKED, CANCELLED, TIMEOUT

# Get execution metrics
metrics = {
    "completed": len(agent.completed_states),
    "running": len(agent._running_states),
    "pending": len(agent.priority_queue)
}

# Track state execution
async def monitored_state(context: Context) -> StateResult:
    # State starts as PENDING
    # Automatically moves to RUNNING when executed
    
    try:
        result = await process_data()
        context.set_state("success", True)
        # State moves to COMPLETED on success
        return "next_state"
    except Exception as e:
        # State moves to FAILED on error
        context.set_state("error", str(e))
        return "handle_error"
```

### 10. Advanced Execution Patterns

#### Conditional Execution
```python
async def route_processing(context: Context) -> StateResult:
    data_size = len(context.get_state("data", []))
    
    if data_size > 1000:
        # Run parallel processing for large data
        return ["process_chunk_1", "process_chunk_2"]
    elif data_size > 100:
        # Run sequential processing for medium data
        return "sequential_process"
    else:
        # Run simple processing for small data
        return "simple_process"
```

#### Dynamic State Generation
```python
# Dynamically add states based on data
data_chunks = split_data(large_data, chunk_size=1000)

for i, chunk in enumerate(data_chunks):
    async def process_chunk(context: Context, chunk_id=i) -> StateResult:
        chunk_data = data_chunks[chunk_id]
        result = await process(chunk_data)
        context.set_state(f"result_{chunk_id}", result)
        return "merge_results"
    
    agent.add_state(
        f"process_chunk_{i}",
        process_chunk,
        resources=ResourceRequirements(cpu_units=1.0)
    )
```

#### State Cleanup and Compensation
```python
async def critical_operation(context: Context) -> StateResult:
    try:
        # Perform operation
        return "next_state"
    except Exception:
        # Add compensation state on failure
        return "cleanup_resources"

async def cleanup_resources(context: Context) -> StateResult:
    # Cleanup any resources
    resources = context.get_state("allocated_resources", [])
    for resource in resources:
        await release_resource(resource)
    return "handle_error"

# Add states with cleanup
agent.add_state("critical_operation", critical_operation)
agent.add_state("cleanup_resources", cleanup_resources)
```

### 11. Complete Example with All Features

```python
from fluffy_agent import (
    Agent, Context, StateResult, 
    ResourceRequirements, Priority,
    DependencyType, RetryPolicy
)
from dataclasses import dataclass
from typing import List, Dict
import time

@dataclass
class ProcessingJob(TypedContextData):
    job_id: str
    data: List[dict]
    config: Dict[str, any]
    timestamp: float

# Create agent with configuration
agent = Agent(
    "complex_workflow",
    max_concurrent=5,
    checkpoint_interval=300
)

# Define states
async def initialize_job(context: Context) -> StateResult:
    # Create job data
    job = ProcessingJob(
        job_id="job_001",
        data=[{"id": i} for i in range(100)],
        config={"batch_size": 10},
        timestamp=time.time()
    )
    context.set_typed("job", job)
    
    # Split into chunks for parallel processing
    chunks = split_data(job.data, job.config["batch_size"])
    for i, chunk in enumerate(chunks):
        context.set_state(f"chunk_{i}", chunk)
    
    return ["process_chunk_0", "process_chunk_1", "process_chunk_2"]

async def process_chunk(context: Context) -> StateResult:
    chunk_id = context.get_state("chunk_id")
    chunk = context.get_state(f"chunk_{chunk_id}")
    
    try:
        result = await process_data(chunk)
        context.set_state(f"result_{chunk_id}", result)
        return "merge_results"
    except Exception as e:
        context.set_state("error", str(e))
        return "handle_error"

async def merge_results(context: Context) -> StateResult:
    # Get original job
    job = context.get_typed("job", ProcessingJob)
    
    # Merge all chunk results
    results = []
    for i in range(10):
        chunk_result = context.get_state(f"result_{i}")
        if chunk_result:
            results.extend(chunk_result)
    
    context.set_state("final_results", results)
    return "human_review"

async def human_review(context: Context) -> StateResult:
    results = context.get_state("final_results")
    
    feedback = await context.human_in_the_loop(
        prompt="Review results. Approve? (yes/no):",
        timeout=300
    )
    
    if "yes" in feedback.lower():
        return "finalize_job"
    else:
        return "handle_rejection"

# Add states with full configuration
agent.add_state(
    "initialize_job",
    initialize_job,
    resources=ResourceRequirements(
        cpu_units=1.0,
        memory_mb=512,
        priority=Priority.HIGH
    )
)

agent.add_state(
    "process_chunk",
    process_chunk,
    resources=ResourceRequirements(
        cpu_units=2.0,
        memory_mb=1024
    ),
    retry_policy=RetryPolicy(
        max_retries=3,
        initial_delay=1.0
    ),
    dependencies={
        "initialize_job": DependencyType.REQUIRED
    }
)

agent.add_state(
    "merge_results",
    merge_results,
    resources=ResourceRequirements(
        cpu_units=2.0,
        memory_mb=2048
    ),
    dependencies={
        "process_chunk": DependencyType.REQUIRED
    }
)

# Run with checkpointing
try:
    await agent.run(timeout=3600)  # 1 hour timeout
except Exception:
    # Save checkpoint on error
    checkpoint = agent.create_checkpoint()
    save_checkpoint(checkpoint)
```

## 🌟 Use Case Examples

### 1. LLM Content Pipeline
```python
@dataclass
class Article(TypedContextData):
    title: str
    content: str
    keywords: List[str]

async def generate_content(context: Context) -> StateResult:
    # Generate with LLM
    article = Article(
        title="AI in 2024",
        content="Generated content...",
        keywords=["AI", "future"]
    )
    context.set_typed("article", article)
    return ["check_quality", "generate_images"]

async def check_quality(context: Context) -> StateResult:
    article = context.get_typed("article", Article)
    if len(article.content) < 1000:
        return "regenerate_content"
    return "await_completion"

# Set up pipeline
agent = Agent("content_generator")
agent.add_state("generate_content", generate_content)
agent.add_state("check_quality", check_quality)
```

### 2. Data Processing Pipeline
```python
@dataclass
class DataBatch(TypedContextData):
    records: List[dict]
    batch_id: str

async def process_batch(context: Context) -> StateResult:
    batch = context.get_typed("batch", DataBatch)
    
    # Split into chunks for parallel processing
    chunks = split_data(batch.records, 3)
    for i, chunk in enumerate(chunks):
        context.set_state(f"chunk_{i}", chunk)
    
    return [f"process_chunk_{i}" for i in range(3)]

async def process_chunk(context: Context) -> StateResult:
    chunk_id = context.get_state("chunk_id")
    data = context.get_state(f"chunk_{chunk_id}")
    # Process chunk...
    return "merge_results"
```

### 3. ML Training Pipeline
```python
@dataclass
class TrainingConfig(TypedContextData):
    model_type: str
    batch_size: int
    epochs: int

async def train_model(context: Context) -> StateResult:
    config = context.get_typed("config", TrainingConfig)
    
    for epoch in range(config.epochs):
        # Train epoch...
        metrics = {"loss": 0.1, "accuracy": 0.95}
        context.set_state(f"metrics_epoch_{epoch}", metrics)
        
        if metrics["accuracy"] > 0.98:
            return "evaluate"
    
    return "evaluate"
```

### 4. Customer Service Automation
```python
@dataclass
class ServiceTicket(TypedContextData):
    id: str
    content: str
    priority: str
    category: str

async def route_ticket(context: Context) -> StateResult:
    ticket = context.get_typed("ticket", ServiceTicket)
    
    if ticket.priority == "high":
        return "urgent_response"
    elif ticket.category == "technical":
        return "technical_review"
    else:
        return "standard_response"

async def urgent_response(context: Context) -> StateResult:
    ticket = context.get_typed("ticket", ServiceTicket)
    # Generate urgent response...
    return "human_review"
```

## 🚀 Quick Start

```python
from fluffy_agent import Agent, Context, StateResult

# Create agent
agent = Agent("my_workflow")

# Define states
async def process_data(context: Context) -> StateResult:
    context.set_state("data", "processed")
    return "format_data"

async def format_data(context: Context) -> StateResult:
    data = context.get_state("data")
    context.set_state("result", data.upper())
    return None

# Add states
agent.add_state("process_data", process_data)
agent.add_state("format_data", format_data)

# Run workflow
await agent.run()
```

## 💡 Key Benefits

- 🎯 Simple, function-based workflow definition
- 🔒 Type safety with flexible data handling
- ⚡ Automatic resource management
- 🔄 Built-in parallel processing
- 🛠️ Comprehensive error handling
- 👥 Human-in-the-loop capabilities
- 📊 Easy scaling from simple to complex workflows

## 📦 Installation

```bash
pip install fluffy-agent-flow
```

## 📚 Learn More

- Full documentation: [docs.fluffy-agent.dev](https://docs.fluffy-agent.dev)
- Examples: [examples/](https://github.com/fluffy-agent/examples)
- Community: [Discord](https://discord.gg/fluffy-agent)
