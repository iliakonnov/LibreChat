const { z } = require('zod');
const axios = require('axios');
const { v4: uuidv4 } = require('uuid');
const { Tool } = require('@langchain/core/tools');
const { tool } = require('@langchain/core/tools');
const { HttpsProxyAgent } = require('https-proxy-agent');
const { FileContext, ContentTypes } = require('librechat-data-provider');
const { logger } = require('~/config');

const displayMessage =
  "Replicate displayed an image. All generated images are already plainly visible, so don't repeat the descriptions in detail. Do not list download links as they are available in the UI already. The user may download the images by clicking on them, but do not mention anything about downloading to the user.";

/**
 * ReplicateAPI - A tool for running models on Replicate.com
 * This tool allows running various models hosted on Replicate, with a focus on image generation.
 */
class ReplicateAPI extends Tool {
  constructor(fields = {}) {
    super();

    /** @type {boolean} Used to initialize the Tool without necessary variables. */
    this.override = fields.override ?? false;

    this.userId = fields.userId;
    this.fileStrategy = fields.fileStrategy;

    /** @type {boolean} **/
    this.isAgent = fields.isAgent;
    this.returnMetadata = fields.returnMetadata ?? false;

    if (fields.processFileURL) {
      /** @type {processFileURL} Necessary for output to contain all image metadata. */
      this.processFileURL = fields.processFileURL.bind(this);
    }

    this.apiKey = fields.REPLICATE_API_KEY || this.getApiKey();

    this.name = 'replicate';
    this.description =
      'Use Replicate to run AI models. This tool can run various models including image generation models.';

    this.description_for_model = `// Use this tool to run AI models on Replicate.com
    // For image generation, provide detailed prompts with visual elements, style, and composition
    // You can specify different models by their identifier (owner/model-name)`;

    // Define the schema for structured input
    this.schema = z.object({
      model: z.string().describe('The model identifier on Replicate (format: "owner/model-name")'),
      input: z.record(z.any()).describe('The input parameters for the model, varies by model type'),
    });
  }

  getAxiosConfig() {
    const config = {};
    if (process.env.PROXY) {
      config.httpsAgent = new HttpsProxyAgent(process.env.PROXY);
    }
    return config;
  }

  /** @param {Object|string} value */
  getDetails(value) {
    if (typeof value === 'string') {
      return value;
    }
    return JSON.stringify(value, null, 2);
  }

  getApiKey() {
    const apiKey = process.env.REPLICATE_API_KEY || '';
    if (!apiKey && !this.override) {
      throw new Error('Missing REPLICATE_API_KEY environment variable.');
    }
    return apiKey;
  }

  wrapInMarkdown(imageUrl) {
    const serverDomain = process.env.DOMAIN_SERVER || 'http://localhost:3080';
    return `![generated image](${serverDomain}${imageUrl})`;
  }

  returnValue(value) {
    if (this.isAgent === true && typeof value === 'string') {
      return [value, {}];
    } else if (this.isAgent === true && typeof value === 'object') {
      if (Array.isArray(value)) {
        return value;
      }
      return [displayMessage, value];
    }
    return value;
  }

  async _call(data) {
    // Use provided API key for this request if available, otherwise use default
    const requestApiKey = this.apiKey || this.getApiKey();

    if (!data.model) {
      throw new Error('Missing required field: model');
    }

    if (!data.input) {
      throw new Error('Missing required field: input');
    }

    const payload = {
      version: data.model,
      input: data.input,
    };

    logger.debug('[ReplicateAPI] Running model with payload:', payload);

    let predictionResponse;
    try {
      predictionResponse = await axios.post('https://api.replicate.com/v1/predictions', payload, {
        headers: {
          Authorization: `Token ${requestApiKey}`,
          'Content-Type': 'application/json',
        },
        ...this.getAxiosConfig(),
      });
    } catch (error) {
      const details = this.getDetails(error?.response?.data || error.message);
      logger.error('[ReplicateAPI] Error while submitting prediction:', details);

      return this.returnValue(
        `Something went wrong when trying to run the model. The Replicate API may be unavailable:
        Error Message: ${details}`,
      );
    }

    const predictionId = predictionResponse.data.id;

    // Polling for the result
    let status = predictionResponse.data.status;
    let output = null;

    while (status !== 'succeeded' && status !== 'failed' && status !== 'canceled') {
      try {
        // Wait 2 seconds between polls
        await new Promise((resolve) => setTimeout(resolve, 2000));

        const statusResponse = await axios.get(
          `https://api.replicate.com/v1/predictions/${predictionId}`,
          {
            headers: {
              Authorization: `Token ${requestApiKey}`,
              'Content-Type': 'application/json',
            },
            ...this.getAxiosConfig(),
          },
        );

        status = statusResponse.data.status;

        if (status === 'succeeded') {
          output = statusResponse.data.output;
          break;
        } else if (status === 'failed' || status === 'canceled') {
          logger.error('[ReplicateAPI] Prediction failed or was canceled:', statusResponse.data);
          return this.returnValue('The model run failed or was canceled.');
        }
      } catch (error) {
        const details = this.getDetails(error?.response?.data || error.message);
        logger.error('[ReplicateAPI] Error while checking prediction status:', details);
        return this.returnValue('An error occurred while checking the model run status.');
      }
    }

    // If no output
    if (!output) {
      logger.error('[ReplicateAPI] No output received from API.');
      return this.returnValue('No output received from Replicate API.');
    }

    // Handle different types of outputs
    // For image outputs (assuming output is a URL or array of URLs)
    if (
      typeof output === 'string' &&
      output.startsWith('http') &&
      (output.endsWith('.png') || output.endsWith('.jpg') || output.endsWith('.jpeg'))
    ) {
      // Handle single image URL by passing it as a singleton array
      return this.handleMultipleImageOutput([output]);
    } else if (
      Array.isArray(output) &&
      output.length > 0 &&
      typeof output[0] === 'string' &&
      output[0].startsWith('http')
    ) {
      // Handle all images in the array
      return this.handleMultipleImageOutput(output);
    } else {
      // For non-image outputs, return as is
      return this.returnValue(`Model output: ${this.getDetails(output)}`);
    }
  }

  async handleMultipleImageOutput(imageUrls) {
    try {
      // Fetch all images and convert to base64
      const fetchOptions = {};
      if (process.env.PROXY) {
        fetchOptions.agent = new HttpsProxyAgent(process.env.PROXY);
      }

      const content = [];
      for (const imageUrl of imageUrls) {
        const imageResponse = await fetch(imageUrl, {
          headers: {
            Authorization: `Token ${this.apiKey || this.getApiKey()}`,
          },
          ...fetchOptions,
        });
        const arrayBuffer = await imageResponse.arrayBuffer();
        const base64 = Buffer.from(arrayBuffer).toString('base64');
        content.push({
          type: ContentTypes.IMAGE_URL,
          image_url: {
            url: `data:image/png;base64,${base64}`,
          },
        });
      }

      const response = [
        {
          type: ContentTypes.TEXT,
          text: displayMessage,
        },
      ];
      return [response, { content }];
    } catch (error) {
      logger.error('Error processing images for agent:', error);
      return this.returnValue(`Failed to process the images. ${error.message}`);
    }
  }
}

/**
 * Creates a tool for querying model input schema from Replicate.com
 * @param {Object} fields - Configuration fields
 * @returns {Tool} - The schema query tool
 */
const createSchemaQueryTool = (fields = {}) => {
  const override = fields.override ?? false;
  const isAgent = fields.isAgent;

  const getApiKey = () => {
    const apiKey = process.env.REPLICATE_API_KEY || '';
    if (!apiKey && !override) {
      throw new Error('Missing REPLICATE_API_KEY environment variable.');
    }
    return apiKey;
  };

  const apiKey = fields.REPLICATE_API_KEY || getApiKey();

  const getAxiosConfig = () => {
    const config = {};
    if (process.env.PROXY) {
      config.httpsAgent = new HttpsProxyAgent(process.env.PROXY);
    }
    return config;
  };

  const returnValue = (value) => {
    if (isAgent === true && typeof value === 'string') {
      return [value, {}];
    } else if (isAgent === true && typeof value === 'object') {
      if (Array.isArray(value)) {
        return value;
      }
      return [value, {}];
    }
    return value;
  };

  return tool(
    async ({ model }) => {
      if (!model) {
        throw new Error('Missing required field: model');
      }

      try {
        const response = await axios.get(`https://api.replicate.com/v1/models/${model}`, {
          headers: {
            Authorization: `Token ${apiKey}`,
            'Content-Type': 'application/json',
          },
          ...getAxiosConfig(),
        });

        const modelData = response.data;

        if (!modelData || !modelData.latest_version || !modelData.latest_version.openapi_schema) {
          return returnValue('No schema information available for this model.');
        }

        const schema = modelData.latest_version.openapi_schema;

        // Extract input schema from OpenAPI schema
        if (
          schema &&
          schema.components &&
          schema.components.schemas &&
          schema.components.schemas.Input
        ) {
          const inputSchema = schema.components.schemas.Input;
          return returnValue(JSON.stringify(inputSchema, null, 2));
        } else {
          return returnValue('Input schema not found in the model information.');
        }
      } catch (error) {
        logger.error('[ReplicateSchemaQuery] Error fetching model schema:', error);
        return returnValue(`Error fetching schema for model: ${error.message}`);
      }
    },
    {
      name: 'replicate_schema',
      description:
        'Query the input schema for a model on Replicate.com to understand what parameters it accepts.',
      schema: z.object({
        model: z
          .string()
          .describe('The model identifier on Replicate (format: "owner/model-name")'),
      }),
      responseFormat: 'content_and_artifact',
    },
  );
};

/**
 * Creates Replicate tools (model execution and schema query)
 * @param {Object} fields - Configuration fields
 * @returns {Array} - Array of Replicate tools
 */
function createReplicateTools(fields = {}) {
  if (!fields.isAgent) {
    throw new Error('This tool is only available for agents.');
  }

  const replicateAPI = new ReplicateAPI(fields);
  const schemaQueryTool = createSchemaQueryTool(fields);

  return [replicateAPI, schemaQueryTool];
}

module.exports = createReplicateTools;
