# 🚀 Agent Framework: Powerful Workflow Orchestration Made Simple

Build robust, scalable workflows as easily as writing Python functions. Perfect for AI/ML pipelines, data processing, and complex automation tasks.

## ✨ Key Features

- **Type-Safe State Management**: Share data between states with full type safety
- **Intelligent Resource Management**: Automatic CPU, memory, and GPU allocation
- **Advanced Flow Control**: Sequential and parallel execution with dependency management
- **Built-in Error Handling**: Retries, timeouts, and graceful failure recovery
- **Checkpointing & Recovery**: Save and restore workflow state at any point
- **Human-in-the-Loop**: Seamlessly integrate human feedback into workflows
- **Resource-Aware Scheduling**: Optimize resource utilization across states

## 🎯 Perfect For

- **AI/ML Engineers**: Orchestrate LLM pipelines and model training workflows
- **Data Engineers**: Build reliable ETL and data processing flows 
- **Backend Developers**: Create resilient service workflows
- **DevOps Engineers**: Automate complex deployment processes
- **Researchers**: Coordinate experiments and analysis pipelines

- ## 💡 Key Benefits

- 🎯 Simple, function-based workflow definition
- 🔒 Type safety with flexible data handling
- ⚡ Automatic resource management
- 🔄 Built-in parallel processing
- 🛠️ Comprehensive error handling
- 👥 Human-in-the-loop capabilities
- 📊 Easy scaling from simple to complex workflows


## 🎓 Quick Start

```python
from agent import Agent, Context, StateResult

# Create your workflow
agent = Agent("hello_workflow")

async def process_data(context: Context) -> StateResult:
    # Process and store data
    result = "Hello, World!"
    context.set_state("greeting", result)
    return "format_output"  # Next state to execute

async def format_output(context: Context) -> StateResult:
    # Get data from previous state
    greeting = context.get_state("greeting")
    print(f"Result: {greeting}")
    return None  # End workflow

# Add states and run
agent.add_state("process_data", process_data)
agent.add_state("format_output", format_output)
await agent.run()
```


## 🔧 Core Features

### 1. State Management System

#### Basic State Operations
```python
async def basic_state(context: Context) -> StateResult:
    # Store any data type
    context.set_state("user_name", "Alice")
    context.set_state("scores", [95, 87, 92])
    
    # Retrieve data
    name = context.get_state("user_name")
    scores = context.get_state("scores", default=[])  # With default value
    
    # Remove state
    context.remove_state("user_name")
    
    # Check state existence
    if context.has_state("scores"):
        print("Scores exist")
        
    return "next_state"
```

#### Type-Safe Data Handling
```python
from dataclasses import dataclass
from typing import List

@dataclass
class UserProfile(TypedContextData):
    name: str
    age: int
    scores: List[float]

async def typed_state(context: Context) -> StateResult:
    # Store typed data
    profile = UserProfile(name="Alice", age=28, scores=[95.0, 87.5])
    context.set_typed("user", profile)
    
    # Update specific fields
    context.update_typed("user", age=29)
    
    # Retrieve with type checking
    user = context.get_typed("user", UserProfile)
    
    return "next_state"
```

### Flow Control

#### Sequential Execution
```python
async def step1(context: Context) -> StateResult:
    return "step2"  # Go to step2

async def step2(context: Context) -> StateResult:
    return "step3"  # Go to step3

async def step3(context: Context) -> StateResult:
    return None  # End workflow
```

#### Parallel Execution
```python
async def start_parallel(context: Context) -> StateResult:
    # Run multiple states in parallel
    return ["process_images", "process_text", "analyze_data"]

async def merge_results(context: Context) -> StateResult:
    # Automatically waits for all parallel states to complete
    return None
```

### Resource Management

#### Resource Requirements
```python
from agent import ResourceRequirements, Priority

# Define detailed resource needs
requirements = ResourceRequirements(
    cpu_units=2.0,        # CPU cores
    memory_mb=1024,       # Memory in MB
    io_weight=1.0,        # I/O priority
    network_weight=1.0,   # Network priority
    gpu_units=1.0,        # GPU units
    priority=Priority.HIGH,
    timeout=300,          # Timeout in seconds
    resource_types=ResourceType.ALL
)

# Add state with resources
agent.add_state(
    "train_model",
    train_model,
    resources=requirements
)
```

#### Resource Pool Configuration
```python
from agent import ResourcePool

# Create custom resource pool
pool = ResourcePool(
    total_cpu=8.0,
    total_memory=16384,  # 16GB
    total_io=100.0,
    total_network=100.0,
    total_gpu=2.0,
    enable_preemption=True,
    enable_quotas=True
)

# Create agent with custom pool
agent = Agent("my_workflow", resource_pool=pool)
```

### Dependency Management

#### Basic Dependencies
```python
agent.add_state(
    "final_step",
    final_step,
    dependencies={
        "prepare_data": DependencyType.REQUIRED,      # Must complete first
        "validate_data": DependencyType.OPTIONAL,     # Run if available
        "background_task": DependencyType.PARALLEL,   # Run in parallel
        "previous_step": DependencyType.SEQUENTIAL    # Run in sequence
    }
)
```

#### Advanced Dependencies
```python
from agent import DependencyConfig, DependencyLifecycle

# Complex dependency configuration
config = DependencyConfig(
    type=DependencyType.REQUIRED,
    lifecycle=DependencyLifecycle.PERIODIC,
    condition=lambda agent: agent.get_state("ready") == True,
    expiry=3600,  # 1 hour
    interval=300,  # 5 minutes
    timeout=60    # 1 minute
)

# Add state with complex dependencies
agent.add_state(
    "complex_step",
    complex_step,
    dependencies={
        "dependency": config
    }
)
```

### Error Handling & Recovery

#### Retry Policies
```python
from agent import RetryPolicy

# Configure retries
retry_policy = RetryPolicy(
    max_retries=3,
    initial_delay=1.0,
    max_delay=60.0,
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

#### Error States and Compensation
```python
async def critical_operation(context: Context) -> StateResult:
    try:
        # Perform operation
        return "success_state"
    except Exception:
        return "handle_error"

async def handle_error(context: Context) -> StateResult:
    # Cleanup and compensation logic
    return "notification_state"

# Add error handling states
agent.add_state("critical_operation", critical_operation)
agent.add_state("handle_error", handle_error)
```

### Checkpointing & Recovery

#### Basic Checkpointing
```python
# Create and save checkpoint
checkpoint = agent.create_checkpoint()
with open("workflow_state.json", "w") as f:
    json.dump(checkpoint.to_dict(), f)

# Restore from checkpoint
new_agent = Agent("restored_workflow")
await new_agent.restore_from_checkpoint(checkpoint)
```

#### Automatic Checkpointing
```python
# Create agent with automatic checkpointing
agent = Agent(
    "my_workflow",
    checkpoint_interval=300  # Checkpoint every 5 minutes
)

# Get checkpoint data
checkpoint_data = {
    "timestamp": checkpoint.timestamp,
    "agent_name": checkpoint.agent_name,
    "status": checkpoint.agent_status,
    "completed": list(checkpoint.completed_states)
}
```

### Human-in-the-Loop Integration

#### Basic Human Input
```python
async def review_content(context: Context) -> StateResult:
    feedback = await context.human_in_the_loop(
        prompt="Review this content (approve/reject):",
        timeout=300  # 5 minute timeout
    )
    
    if "approve" in feedback.lower():
        return "publish"
    else:
        return "revise"
```

#### Validated Human Input
```python
async def get_user_input(context: Context) -> StateResult:
    # Get input with validation
    score = await context.human_in_the_loop(
        prompt="Enter score (0-100):",
        timeout=60,
        validator=lambda x: x.isdigit() and 0 <= int(x) <= 100,
        default="50"  # Default if timeout
    )
    
    context.set_state("user_score", int(score))
    return "process_score"
```

### Monitoring & Metrics

#### State Status Tracking
```python
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
```

#### Resource Usage Monitoring
```python
# Get resource usage stats
stats = agent.resource_pool.get_usage_stats()
print(f"CPU Usage: {stats[ResourceType.CPU].current_usage}")
print(f"Memory Usage: {stats[ResourceType.MEMORY].current_usage}")

# Get allocations by state
allocations = agent.resource_pool.get_state_allocations()
waiting_states = agent.resource_pool.get_waiting_states()
```

### Advanced Execution Control

#### Execution Configuration
```python
# Run with timeout
await agent.run(timeout=3600)  # 1 hour timeout

# Run with cleanup handling
await agent.run(cleanup_timeout=60)  # 1 minute cleanup

# Configure maximum concurrent states
agent = Agent(
    "my_workflow",
    max_concurrent=5  # Max 5 states running at once
)
```

#### Execution Control
```python
# Pause execution
checkpoint = await agent.pause()

# Resume execution
await agent.resume()

# Cancel specific state
agent.cancel_state("state_name")

# Cancel all states
await agent.cancel_all()
```

### Dynamic Workflows

#### Dynamic State Generation
```python
# Generate states based on data
data_chunks = split_data(large_data, chunk_size=1000)

for i, chunk in enumerate(data_chunks):
    async def process_chunk(context: Context, chunk_id=i) -> StateResult:
        result = await process(data_chunks[chunk_id])
        context.set_state(f"result_{chunk_id}", result)
        return "merge_results"
    
    agent.add_state(
        f"process_chunk_{i}",
        process_chunk,
        resources=ResourceRequirements(cpu_units=1.0)
    )
```

#### Conditional Workflows
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

## 📊 Monitoring & Metrics

Track, analyze, and optimize your workflows with comprehensive monitoring capabilities.

### Real-Time Monitoring

#### Basic Monitoring
```python
from agent import Agent
from agent_monitor import agent_monitor, MetricType

# Enable monitoring with decorator
@agent_monitor(
    metrics=MetricType.ALL,  # Track all metric types
    realtime=True,          # Enable real-time updates
    update_interval=1.0     # Update every second
)
async def run_workflow():
    agent = Agent("monitored_workflow")
    # Add states and run...
    await agent.run()

# Custom metrics callback
async def handle_metrics(metrics: dict):
    print(f"Current throughput: {metrics['throughput']['states_per_second']['avg']}")
    print(f"Resource usage: {metrics['resources']['used_CPU']['current']}")

# Run with metrics handling
@agent_monitor(metrics_callback=handle_metrics)
async def run_monitored():
    await run_workflow()
```

#### Selective Metric Collection
```python
# Monitor specific metrics
@agent_monitor(
    metrics=(
        MetricType.TIMING |      # Track execution times
        MetricType.RESOURCES |   # Monitor resource usage
        MetricType.THROUGHPUT    # Measure operations/second
    ),
    aggregate=True  # Get aggregated statistics
)
async def run_optimized():
    agent = Agent("optimized_workflow")
    # Add states and run...
    await agent.run()
```

### Available Metrics

1. **Timing Metrics**
   - Total execution time
   - State-level timing
   - Dependency resolution timing

2. **Resource Usage**
   - CPU utilization
   - Memory consumption
   - I/O operations
   - GPU usage (if applicable)

3. **State Metrics**
   - State transitions
   - Success/failure rates
   - Retry statistics

4. **Throughput Analysis**
   - States per second
   - Queue processing rate
   - Concurrent operations

5. **Dependency Metrics**
   - Resolution time
   - Dependency chain length
   - Blocking dependencies

### Advanced Monitoring

#### Custom Metric Aggregation
```python
from agent_monitor import MetricAggregation

# Create custom metric handler
class CustomMetrics:
    def __init__(self):
        self.metrics = defaultdict(MetricAggregation)
    
    async def handle_metrics(self, data: dict):
        for category, metrics in data.items():
            for name, values in metrics.items():
                self.metrics[f"{category}_{name}"].add(values['avg'])
                
        # Calculate custom KPIs
        throughput = self.metrics['throughput_states_per_second'].avg
        cpu_usage = self.metrics['resources_used_CPU'].avg
        efficiency = throughput / cpu_usage if cpu_usage > 0 else 0
        print(f"Workflow efficiency: {efficiency:.2f}")

# Use custom metrics handler
metrics_handler = CustomMetrics()
@agent_monitor(
    metrics_callback=metrics_handler.handle_metrics,
    realtime=True
)
async def run_with_custom_metrics():
    # Run workflow...
    pass
```

#### Performance Analysis
```python
# Detailed performance monitoring
@agent_monitor(
    metrics=(
        MetricType.TIMING |
        MetricType.THROUGHPUT |
        MetricType.LATENCY
    )
)
async def analyze_performance():
    async def process_metrics(data: dict):
        # Extract performance metrics
        latency = data['timing']['total_execution']['p95']
        throughput = data['throughput']['states_per_second']['avg']
        
        # Log performance data
        logging.info(f"P95 Latency: {latency}ms")
        logging.info(f"Avg Throughput: {throughput} states/sec")
        
        # Check performance thresholds
        if latency > 1000:  # 1 second
            logging.warning("High latency detected")
        if throughput < 10:
            logging.warning("Low throughput detected")
    
    agent = Agent("performance_workflow")
    # Add states and run...
    await agent.run()
```

### Best Practices

1. **Resource Monitoring**
   - Monitor resource usage in real-time
   - Set alerts for resource thresholds
   - Track resource efficiency metrics

2. **Performance Optimization**
   - Use metrics to identify bottlenecks
   - Monitor state execution patterns
   - Analyze dependency resolution times

3. **Metric Collection**
   - Choose relevant metric types
   - Set appropriate update intervals
   - Use aggregation for long-running workflows

4. **Error Tracking**
   - Monitor error rates and patterns
   - Track retry statistics
   - Analyze failure points

### Visualization Integration

#### Metrics Dashboard Example
```python
from agent_monitor import RealTimeMetrics

# Create real-time metrics dashboard
class MetricsDashboard:
    def __init__(self):
        self.rt_metrics = RealTimeMetrics(update_interval=1.0)
        
    async def update_dashboard(self, metrics: dict):
        # Update dashboard with latest metrics
        throughput = metrics['throughput']['states_per_second']['avg']
        errors = metrics['errors']['count']['sum']
        cpu_usage = metrics['resources']['used_CPU']['current']
        
        print("\033[2J\033[H")  # Clear screen
        print("=== Workflow Dashboard ===")
        print(f"Throughput: {throughput:.2f} states/sec")
        print(f"Errors: {errors}")
        print(f"CPU Usage: {cpu_usage:.1f}%")
        
    async def start(self):
        self.rt_metrics.subscribe(self.update_dashboard)
        self.rt_metrics.start()
        
    async def stop(self):
        self.rt_metrics.stop()

# Use dashboard in workflow
dashboard = MetricsDashboard()
@agent_monitor(
    metrics_callback=dashboard.update_dashboard,
    realtime=True
)
async def run_with_dashboard():
    await dashboard.start()
    try:
        # Run workflow...
        pass
    finally:
        await dashboard.stop()
```

## 🔄 Coordination & Control

Scale your workflows with production-grade coordination primitives, rate limiting, and deadlock prevention.

### Basic Coordination

#### Create Coordinated Agent
```python
from agent import Agent
from agent.coordination import create_coordinated_agent, PrimitiveType, RateLimitStrategy

# Create agent with coordination enabled
agent = create_coordinated_agent("coordinated_workflow")

# Or enhance existing agent
from agent.coordination import enhance_agent
agent = enhance_agent(Agent("my_workflow"))
```

#### Rate Limiting
```python
# Add rate limit to specific state
agent.add_state_rate_limit(
    "api_call",
    max_rate=10.0,          # 10 calls per second
    strategy=RateLimitStrategy.TOKEN_BUCKET,
    burst_size=5            # Allow bursts of 5 calls
)

# Add state with rate limiting
async def api_call(context: Context) -> StateResult:
    # Rate limiting handled automatically
    return "next_state"

agent.add_state("api_call", api_call)
```

### Coordination Primitives

#### Mutex (Exclusive Access)
```python
# Add mutex for resource access
agent.add_state_coordination(
    "database_write",
    PrimitiveType.MUTEX,
    ttl=30.0  # Auto-release after 30 seconds
)

async def write_to_db(context: Context) -> StateResult:
    # Mutex ensures exclusive access
    data = context.get_state("data")
    # Write data...
    return "next_state"

agent.add_state("database_write", write_to_db)
```

#### Semaphore (Concurrent Access)
```python
# Limit concurrent operations
agent.add_state_coordination(
    "heavy_process",
    PrimitiveType.SEMAPHORE,
    max_count=3  # Allow 3 concurrent executions
)

async def process_batch(context: Context) -> StateResult:
    # Semaphore controls concurrency
    return "next_state"

agent.add_state("heavy_process", process_batch)
```

#### Resource Quotas
```python
# Add quota management
agent.add_state_coordination(
    "gpu_task",
    PrimitiveType.QUOTA,
    quota_limit=16.0,  # Total GPU memory limit
    ttl=3600.0        # Reset quota hourly
)

async def train_model(context: Context) -> StateResult:
    # Quota ensures resource availability
    return "next_state"

agent.add_state("gpu_task", train_model)
```

### Advanced Coordination

#### Complex Rate Limiting
```python
from agent.coordination import RateLimitStrategy

# Sliding window rate limit
agent.add_state_rate_limit(
    "api_call",
    max_rate=100.0,
    strategy=RateLimitStrategy.SLIDING_WINDOW,
    window_size=60.0  # 1 minute window
)

# Leaky bucket for smooth flow
agent.add_state_rate_limit(
    "data_ingestion",
    max_rate=50.0,
    strategy=RateLimitStrategy.LEAKY_BUCKET,
    window_size=1.0
)
```

#### Barrier Synchronization
```python
# Create synchronization point
agent.add_state_coordination(
    "merge_results",
    PrimitiveType.BARRIER,
    max_count=5,        # Wait for 5 states
    wait_timeout=30.0   # Timeout after 30 seconds
)

async def process_chunk(context: Context) -> StateResult:
    # Process data chunk
    return "merge_results"

async def merge_results(context: Context) -> StateResult:
    # Executes when all chunks complete
    return "next_state"

# Add parallel states
for i in range(5):
    agent.add_state(f"process_chunk_{i}", process_chunk)
agent.add_state("merge_results", merge_results)
```

#### Lease Management
```python
# Time-based exclusive access
agent.add_state_coordination(
    "batch_process",
    PrimitiveType.LEASE,
    ttl=600.0  # 10 minute lease
)

async def process_batch(context: Context) -> StateResult:
    # Lease ensures single execution
    return "next_state"

agent.add_state("batch_process", process_batch)
```

### Deadlock Prevention

#### Automatic Detection
```python
# Configure deadlock detection
agent._coordinator.deadlock_detector.detection_interval = 1.0
agent._coordinator.deadlock_detector.max_cycles = 100

# Get detection status
status = await agent.get_coordination_status()
if status["deadlock_detector"]["cycle_count"] > 0:
    print("Deadlock detected!")
```

#### Resource Cleanup
```python
# Configure cleanup
agent._coordinator.cleanup_interval = 60.0  # 1 minute

# Manual cleanup
async def cleanup_resources(context: Context) -> StateResult:
    await agent.reset_coordination()
    return "next_state"
```

### Coordination Monitoring

#### Status Tracking
```python
async def monitor_coordination(context: Context) -> StateResult:
    status = await agent.get_coordination_status()
    
    # Check primitive states
    for name, state in status["primitives"].items():
        print(f"Primitive {name}: {state['state']}")
        print(f"Active owners: {len(state['owners'])}")
        print(f"Wait count: {state['wait_count']}")
    
    # Check deadlock detector
    detector = status["deadlock_detector"]
    print(f"Detected cycles: {detector['cycle_count']}")
    
    return "next_state"
```

### Best Practices

1. **Rate Limiting**
   - Match rate limits to downstream capacity
   - Use appropriate strategies for workload
   - Set reasonable burst allowances

2. **Resource Management**
   - Set appropriate TTLs for primitives
   - Use quotas for finite resources
   - Monitor resource usage patterns

3. **Deadlock Prevention**
   - Enable automatic detection
   - Set reasonable detection intervals
   - Plan recovery strategies

4. **Error Handling**
   - Handle coordination timeouts
   - Implement retry strategies
   - Clean up resources properly

5. **Performance**
   - Use appropriate primitive types
   - Monitor coordination overhead
   - Balance safety vs. performance

### Production Patterns

#### Distributed Processing
```python
# Coordinate distributed workers
agent.add_state_coordination(
    "distributed_task",
    PrimitiveType.SEMAPHORE,
    max_count=10,
    ttl=3600.0
)

agent.add_state_rate_limit(
    "distributed_task",
    max_rate=100.0,
    strategy=RateLimitStrategy.SLIDING_WINDOW,
    window_size=60.0
)

async def process_distributed(context: Context) -> StateResult:
    # Coordinated processing across workers
    chunk = context.get_state("chunk")
    results = await process_chunk(chunk)
    context.set_state(f"results_{chunk_id}", results)
    return "merge_results"
```

#### Resource Pools
```python
# Manage shared resource pool
agent.add_state_coordination(
    "shared_resource",
    PrimitiveType.QUOTA,
    quota_limit=100.0,
    ttl=60.0
)

async def use_resource(context: Context) -> StateResult:
    # Get resource allocation
    allocation = 10.0
    async with ResourceAllocation("shared_resource", allocation):
        # Use resource...
        pass
    return "next_state"
```

This coordination system provides robust control over workflow execution, resource usage, and concurrency, enabling safe and efficient operation at scale.

## 📄 License

MIT License - Free for personal and commercial use.

---

Built with ❤️ by the Agent Framework Team
- Examples: [examples/](https://github.com/fluffy-agent/examples)
- Community: [Discord](https://discord.gg/dCtExd)
