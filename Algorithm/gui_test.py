import tkinter as tk

class PointSelector:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Selector")
        self.canvas = tk.Canvas(root, width=500, height=500, bg="white")
        self.canvas.pack()

        self.points = []  # List to store the selected points
        self.circles = []  # Store circle IDs for the points
        self.dragging_point = None  # Track which point is being dragged

        # Bind events
        self.canvas.bind("<Button-1>", self.select_point)
        self.canvas.bind("<B1-Motion>", self.drag_point)
        self.canvas.bind("<ButtonRelease-1>", self.stop_dragging)

    def select_point(self, event):
        if len(self.points) < 2:
            # Add a new point
            self.points.append((event.x, event.y))
            r = 5  # Radius of the circle
            circle = self.canvas.create_oval(event.x - r, event.y - r, event.x + r, event.y + r, fill="red")
            self.circles.append(circle)
            print(f"Point {len(self.points)}: ({event.x}, {event.y})")
        else:
            # Check if clicking near an existing point
            for i, (x, y) in enumerate(self.points):
                if abs(event.x - x) <= 5 and abs(event.y - y) <= 5:
                    self.dragging_point = i
                    break

    def drag_point(self, event):
        if self.dragging_point is not None:
            # Update the point position
            idx = self.dragging_point
            self.points[idx] = (event.x, event.y)
            print(f"Dragging Point {idx + 1}: ({event.x}, {event.y})")

            # Move the corresponding circle
            r = 5
            self.canvas.coords(self.circles[idx], event.x - r, event.y - r, event.x + r, event.y + r)

    def stop_dragging(self, event):
        if self.dragging_point is not None:
            print(f"Point {self.dragging_point + 1} finalized at: {self.points[self.dragging_point]}")
        self.dragging_point = None

# Create the application window
if __name__ == "__main__":
    root = tk.Tk()
    app = PointSelector(root)
    root.mainloop()
