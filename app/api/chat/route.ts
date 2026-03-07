import { createResource } from '@/lib/actions/resources';
import {
  convertToModelMessages,
  streamText,
  tool,
  UIMessage,
  stepCountIs,
} from 'ai';
import { z } from 'zod';
import { findRelevantContent } from '@/lib/ai/embedding';

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

// Zod schema for validating UIMessage (basic check for role + content)
const uiMessageSchema = z.object({
  role: z.enum(['user', 'assistant', 'system']),
  content: z.string().min(1),
  // Add more fields if needed (e.g., toolCalls, etc.)
});

export async function POST(req: Request) {
  let messages: UIMessage[];
  try {
    const body = await req.json();
    const parsedMessages = z.array(uiMessageSchema).safeParse(body?.messages);
    if (!parsedMessages.success) {
      return new Response(JSON.stringify({ error: `Invalid request body: ${parsedMessages.error.message}`, code: 'INVALID_BODY' }), {
        status: 400,
        headers: { 'Content-Type': 'application/json' },
      });
    }
    messages = parsedMessages.data;
  } catch {
    return new Response(JSON.stringify({ error: 'Invalid request body', code: 'INVALID_JSON' }), {
      status: 400,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  let result;
  try {
    result = streamText({
      model: 'openai/gpt-4o',
      messages: convertToModelMessages(messages),
      stopWhen: stepCountIs(10),
      system: `You are a helpful assistant with access to a personal knowledge base.
Prefer using the getInformation tool to retrieve relevant context before answering, but use general knowledge if appropriate.
If the user shares a new piece of knowledge, use the addResource tool to save it immediately without asking for confirmation.
If no relevant information is found in the knowledge base and general knowledge isn't sufficient, respond with: "Sorry, I don't know."`,
      tools: {
        addResource: tool({
          description: `add a resource to your knowledge base.
          If the user provides a random piece of knowledge unprompted, use this tool without asking for confirmation.`,
          inputSchema: z.object({
            content: z
              .string()
              .describe('the content or resource to add to the knowledge base'),
          }),
          execute: async ({ content }) => {
            try {
              const result = await createResource({ content });
              return { success: true, result };
            } catch (error) {
              console.error('addResource error:', error);
              return { success: false, error: 'Failed to add resource' };
            }
          },
        }),
        getInformation: tool({
          description: `get information from your knowledge base to answer questions.`,
          inputSchema: z.object({
            question: z.string().describe('the users question'),
          }),
          execute: async ({ question }) => {
            try {
              const results = await findRelevantContent(question);
              return results; // Assuming this returns an array or object
            } catch (error) {
              console.error('getInformation error:', error);
              return []; // Return empty array to indicate no results
            }
          },
        }),
      },
    });
  } catch (error) {
    console.error('streamText error:', error);
    return new Response(JSON.stringify({ error: 'Internal server error', code: 'STREAM_ERROR' }), {
      status: 500,
      headers: { 'Content-Type': 'application/json' },
    });
  }

  return result.toUIMessageStreamResponse();
}