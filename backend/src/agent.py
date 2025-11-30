import logging
import json
import os
from datetime import datetime
from typing import Optional
from pathlib import Path

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentSession,
    JobContext,
    JobProcess,
    MetricsCollectedEvent,
    RoomInputOptions,
    WorkerOptions,
    cli,
    metrics,
    tokenize,
    function_tool,
    RunContext
)
from livekit.plugins import murf, silero, google, deepgram, noise_cancellation
from livekit.plugins.turn_detector.multilingual import MultilingualModel

logger = logging.getLogger("agent")

load_dotenv(".env.local")

# File paths
DB_DIR = Path("DB")
CATALOG_FILE = DB_DIR / "catalog.json"
ORDERS_FILE = DB_DIR / "orders.json"

# Ensure DB directory exists
DB_DIR.mkdir(exist_ok=True)


def load_catalog() -> list[dict]:
    """Load product catalog from DB/catalog.json"""
    try:
        if CATALOG_FILE.exists():
            with open(CATALOG_FILE, 'r', encoding='utf-8') as f:
                catalog = json.load(f)
                logger.info(f"Loaded {len(catalog)} products from catalog")
                return catalog
        else:
            logger.warning(f"Catalog file not found at {CATALOG_FILE}")
            return []
    except Exception as e:
        logger.error(f"Error loading catalog: {e}")
        return []


def load_orders() -> list[dict]:
    """Load orders from DB/orders.json"""
    try:
        if ORDERS_FILE.exists():
            with open(ORDERS_FILE, 'r', encoding='utf-8') as f:
                orders = json.load(f)
                logger.info(f"Loaded {len(orders)} orders")
                return orders
        else:
            logger.info("No existing orders file, starting fresh")
            return []
    except Exception as e:
        logger.error(f"Error loading orders: {e}")
        return []


def save_orders(orders: list[dict]) -> bool:
    """Save orders to DB/orders.json"""
    try:
        with open(ORDERS_FILE, 'w', encoding='utf-8') as f:
            json.dump(orders, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(orders)} orders to file")
        return True
    except Exception as e:
        logger.error(f"Error saving orders: {e}")
        return False


def list_products(
    category: Optional[str] = None,
    max_price: Optional[int] = None,
    color: Optional[str] = None,
    search_term: Optional[str] = None
) -> list[dict]:
    """
    Filter and return products from the catalog.
    
    Args:
        category: Filter by product category (e.g., 'mug', 'tshirt', 'hoodie')
        max_price: Maximum price in INR
        color: Filter by color
        search_term: Search in product name or description
    
    Returns:
        List of matching products
    """
    products = load_catalog()
    results = products.copy()
    
    if category:
        results = [p for p in results if p.get("category", "").lower() == category.lower()]
    
    if max_price:
        results = [p for p in results if p["price"] <= max_price]
    
    if color:
        results = [p for p in results if p.get("color", "").lower() == color.lower()]
    
    if search_term:
        search_lower = search_term.lower()
        results = [
            p for p in results 
            if search_lower in p["name"].lower() or search_lower in p.get("description", "").lower()
        ]
    
    return results


def create_order(line_items: list[dict]) -> dict:
    """
    Create a new order from line items.
    
    Args:
        line_items: List of dicts with keys: product_id, quantity, and optionally size
        
    Returns:
        Order object with id, items, total, currency, and timestamp
    """
    products = load_catalog()
    orders = load_orders()
    
    order_items = []
    total = 0
    
    for item in line_items:
        product_id = item.get("product_id")
        quantity = item.get("quantity", 1)
        size = item.get("size")
        
        # Find the product
        product = next((p for p in products if p["id"] == product_id), None)
        
        if not product:
            raise ValueError(f"Product {product_id} not found")
        
        if not product.get("in_stock", False):
            raise ValueError(f"Product {product['name']} is out of stock")
        
        # Validate size if applicable
        if size and "sizes" in product:
            if size.upper() not in product["sizes"]:
                raise ValueError(f"Size {size} not available for {product['name']}")
        
        item_total = product["price"] * quantity
        total += item_total
        
        order_items.append({
            "product_id": product_id,
            "product_name": product["name"],
            "quantity": quantity,
            "size": size,
            "unit_price": product["price"],
            "item_total": item_total
        })
    
    # Generate order
    order = {
        "id": f"ORD-{len(orders) + 1:04d}",
        "items": order_items,
        "total": total,
        "currency": "INR",
        "created_at": datetime.now().isoformat(),
        "status": "confirmed"
    }
    
    # Add to orders list and save
    orders.append(order)
    save_orders(orders)
    
    # Log order for debugging
    logger.info(f"Order created: {json.dumps(order, indent=2)}")
    
    return order


def get_last_order() -> Optional[dict]:
    """
    Retrieve the most recent order.
    
    Returns:
        The last order object or None if no orders exist
    """
    orders = load_orders()
    if orders:
        return orders[-1]
    return None


def get_all_orders() -> list[dict]:
    """
    Retrieve all orders.
    
    Returns:
        List of all order objects
    """
    return load_orders()


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a helpful voice shopping assistant for an online store. The user is interacting with you via voice.

You help customers:
- Browse our product catalog (mugs, t-shirts, hoodies, and more)
- Find products that match their preferences
- Answer questions about specific products
- Place orders with proper confirmation
- Review their orders

Your responses are conversational, friendly, and concise. When describing products, mention the name, price in rupees, and key features. When taking orders, always confirm the product, quantity, and size (if applicable) before creating the order.

Remember:
- Prices are in Indian Rupees (INR)
- Some products have size options (S, M, L, XL)
- Always confirm order details before finalizing
- Be helpful and enthusiastic about our products
- You can access the complete catalog to answer any product questions""",
        )
        # Store conversation context for product references
        self.last_products_shown = []

    @function_tool
    async def browse_products(
        self,
        context: RunContext,
        category: Optional[str] = None,
        max_price: Optional[int] = None,
        color: Optional[str] = None,
        search_term: Optional[str] = None
    ):
        """Browse the product catalog with optional filters.
        
        Use this tool when the customer wants to see products or is looking for something specific.
        
        Args:
            category: Product category like 'mug', 'tshirt', or 'hoodie'
            max_price: Maximum price in rupees (INR)
            color: Color preference like 'black', 'white', 'blue', 'gray'
            search_term: Search text to match in product name or description
        """
        logger.info(f"Browsing products: category={category}, max_price={max_price}, color={color}, search={search_term}")
        
        products = list_products(
            category=category,
            max_price=max_price,
            color=color,
            search_term=search_term
        )
        
        # Store for reference
        self.last_products_shown = products
        
        if not products:
            return "No products found matching those criteria."
        
        # Format product list
        result = f"Found {len(products)} product(s):\n"
        for i, p in enumerate(products, 1):
            result += f"{i}. {p['name']} - {p['price']} rupees"
            if 'color' in p:
                result += f" ({p['color']})"
            if 'sizes' in p:
                result += f" - Available in sizes: {', '.join(p['sizes'])}"
            result += f"\n   {p['description']}\n"
        
        return result

    @function_tool
    async def place_order(
        self,
        context: RunContext,
        product_id: str,
        quantity: int = 1,
        size: Optional[str] = None
    ):
        """Place an order for a product.
        
        Use this tool when the customer confirms they want to buy a product.
        Always confirm the product details with the customer before calling this.
        
        Args:
            product_id: The product ID (like 'mug-001', 'tshirt-001', 'hoodie-002')
            quantity: Number of items to order (default: 1)
            size: Size for clothing items (S, M, L, XL) if applicable
        """
        logger.info(f"Creating order: product_id={product_id}, quantity={quantity}, size={size}")
        
        try:
            line_items = [{
                "product_id": product_id,
                "quantity": quantity,
                "size": size
            }]
            
            order = create_order(line_items)
            
            # Format order confirmation
            result = f"Order confirmed! Order ID: {order['id']}\n"
            result += "Items:\n"
            for item in order['items']:
                result += f"- {item['product_name']} x {item['quantity']}"
                if item.get('size'):
                    result += f" (Size: {item['size']})"
                result += f" - {item['item_total']} rupees\n"
            result += f"Total: {order['total']} rupees\n"
            result += f"Your order has been placed successfully and saved!"
            
            return result
            
        except ValueError as e:
            return f"Sorry, I couldn't place that order: {str(e)}"
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            return "Sorry, there was an error placing your order. Please try again."

    @function_tool
    async def view_last_order(self, context: RunContext):
        """View the most recent order.
        
        Use this tool when the customer asks what they just bought or wants to review their last order.
        """
        logger.info("Retrieving last order")
        
        order = get_last_order()
        
        if not order:
            return "You haven't placed any orders yet."
        
        result = f"Your last order (ID: {order['id']}):\n"
        result += "Items:\n"
        for item in order['items']:
            result += f"- {item['product_name']} x {item['quantity']}"
            if item.get('size'):
                result += f" (Size: {item['size']})"
            result += f" - {item['item_total']} rupees\n"
        result += f"Total: {order['total']} rupees\n"
        result += f"Status: {order['status']}"
        
        return result

    @function_tool
    async def view_all_orders(self, context: RunContext):
        """View all orders that have been placed.
        
        Use this tool when the customer asks to see their order history or all orders.
        """
        logger.info("Retrieving all orders")
        
        orders = get_all_orders()
        
        if not orders:
            return "No orders have been placed yet."
        
        result = f"Order History ({len(orders)} order(s)):\n\n"
        for order in orders:
            result += f"Order {order['id']} - {order['created_at'][:10]}\n"
            for item in order['items']:
                result += f"  - {item['product_name']} x {item['quantity']}"
                if item.get('size'):
                    result += f" (Size: {item['size']})"
                result += "\n"
            result += f"  Total: {order['total']} rupees\n\n"
        
        return result

    @function_tool
    async def get_product_details(self, context: RunContext, product_reference: str):
        """Get detailed information about a specific product.
        
        Use this when the customer asks about a specific product by name or reference
        (like "the second hoodie" or "that black mug").
        
        Args:
            product_reference: Product name, ID, or description reference
        """
        logger.info(f"Getting product details for: {product_reference}")
        
        products = load_catalog()
        
        # Try to find by ID first
        product = next((p for p in products if p["id"] == product_reference), None)
        
        # If not found, try searching by name
        if not product:
            search_results = list_products(search_term=product_reference)
            if search_results:
                product = search_results[0]
        
        if not product:
            return f"I couldn't find a product matching '{product_reference}'. Would you like me to show you our catalog?"
        
        result = f"{product['name']}\n"
        result += f"Price: {product['price']} rupees\n"
        result += f"Description: {product['description']}\n"
        if 'color' in product:
            result += f"Color: {product['color']}\n"
        if 'sizes' in product:
            result += f"Available sizes: {', '.join(product['sizes'])}\n"
        result += f"Product ID: {product['id']}\n"
        result += f"In stock: {'Yes' if product.get('in_stock') else 'No'}"
        
        return result


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    # Logging setup
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Initialize catalog and orders files if they don't exist
    if not CATALOG_FILE.exists():
        logger.warning("Creating sample catalog file...")
        sample_catalog = [
            {
                "id": "mug-001",
                "name": "Stoneware Coffee Mug",
                "description": "Handcrafted ceramic mug perfect for your morning coffee",
                "price": 800,
                "currency": "INR",
                "category": "mug",
                "color": "white",
                "in_stock": True
            },
            {
                "id": "mug-002",
                "name": "Blue Ceramic Mug",
                "description": "Classic blue glazed ceramic mug",
                "price": 650,
                "currency": "INR",
                "category": "mug",
                "color": "blue",
                "in_stock": True
            },
            {
                "id": "tshirt-001",
                "name": "Cotton T-Shirt",
                "description": "Comfortable cotton t-shirt for everyday wear",
                "price": 899,
                "currency": "INR",
                "category": "tshirt",
                "color": "black",
                "sizes": ["S", "M", "L", "XL"],
                "in_stock": True
            },
            {
                "id": "tshirt-002",
                "name": "Premium White Tee",
                "description": "Premium quality white cotton t-shirt",
                "price": 1200,
                "currency": "INR",
                "category": "tshirt",
                "color": "white",
                "sizes": ["S", "M", "L", "XL"],
                "in_stock": True
            },
            {
               "id": "cap-001",
               "name": "Baseball Cap",
               "description": "Adjustable cotton baseball cap",
               "price": 349,
               "currency": "INR",
               "category": "accessories",
               "color": "black",
               "sizes": ["S", "M"],
               "in_stock": True
            },
            {
                "id": "hoodie-001",
                "name": "Black Hoodie",
                "description": "Warm and cozy black hoodie with front pocket",
                "price": 2499,
                "currency": "INR",
                "category": "hoodie",
                "color": "black",
                "sizes": ["M", "L", "XL"],
                "in_stock": True
            },
            {
                "id": "hoodie-002",
                "name": "Gray Zip Hoodie",
                "description": "Premium gray hoodie with zipper",
                "price": 2799,
                "currency": "INR",
                "category": "hoodie",
                "color": "gray",
                "sizes": ["S", "M", "L", "XL"],
                "in_stock": True
            },
        ]
        with open(CATALOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(sample_catalog, f, indent=2, ensure_ascii=False)
        logger.info(f"Sample catalog created at {CATALOG_FILE}")
    
    if not ORDERS_FILE.exists():
        logger.info(f"Initializing empty orders file at {ORDERS_FILE}")
        with open(ORDERS_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f)

    # Set up voice AI pipeline
    session = AgentSession(
        stt=deepgram.STT(model="nova-3"),
        llm=google.LLM(model="gemini-2.5-flash"),
        tts=murf.TTS(
            voice="en-US-matthew", 
            style="Conversation",
            tokenizer=tokenize.basic.SentenceTokenizer(min_sentence_len=2),
            text_pacing=True
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    # Metrics collection
    usage_collector = metrics.UsageCollector()

    @session.on("metrics_collected")
    def _on_metrics_collected(ev: MetricsCollectedEvent):
        metrics.log_metrics(ev.metrics)
        usage_collector.collect(ev.metrics)

    async def log_usage():
        summary = usage_collector.get_summary()
        logger.info(f"Usage: {summary}")

    ctx.add_shutdown_callback(log_usage)

    # Start the session
    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, prewarm_fnc=prewarm))