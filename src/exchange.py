class Exchange:

    def __init__(self):
        self.price_to_orders = {}
        self.orders_to_price = {}
        raise NotImplementedError

    def handle_order(self, order):
        raise NotImplementedError