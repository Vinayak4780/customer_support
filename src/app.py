import gradio as gr
from pipeline import predict_ticket, get_product_list

# Paths to models and data
DATA_PATH = 'ai_dev_assignment_tickets_complex_1000.xls'
MODEL_ISSUE_PATH = 'src/issue_type_model.pkl'
MODEL_URGENCY_PATH = 'src/urgency_level_model.pkl'

# Load product list once
product_list = get_product_list(DATA_PATH)

def gradio_predict(ticket_text):
    result = predict_ticket(ticket_text, product_list, MODEL_ISSUE_PATH, MODEL_URGENCY_PATH)
    # Return outputs in the order: issue type, urgency, entities
    return result['predicted_issue_type'], result['predicted_urgency_level'], result['entities']

iface = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(lines=6, label="Enter Ticket Text"),
    outputs=[
        gr.Textbox(label="Predicted Issue Type"),
        gr.Textbox(label="Predicted Urgency Level"),
        gr.JSON(label="Extracted Entities")
    ],
    title="Customer Support Ticket Classifier & Entity Extractor",
    description="Paste a customer support ticket to get predictions and extracted entities."
)

if __name__ == "__main__":
    iface.launch()
