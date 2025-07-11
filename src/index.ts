import { AppServer, AppSession, ViewType, AuthenticatedRequest, PhotoData } from '@mentra/sdk';
import { Request, Response } from 'express';
import * as ejs from 'ejs';
import * as path from 'path';
import axios from 'axios';

/**
 * Interface representing a stored photo with metadata
 */
interface StoredPhoto {
  requestId: string;
  buffer: Buffer;
  timestamp: Date;
  userId: string;
  mimeType: string;
  filename: string;
  size: number;
}

/**
 * Interface for face detection prediction from Roboflow API
 */
interface FacePrediction {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  class: string;
  class_id: number;
  detection_id: string;
}

/**
 * Interface for face detection response from Roboflow API
 */
interface FaceDetectionResponse {
  predictions: FacePrediction[];
}

const PACKAGE_NAME = process.env.PACKAGE_NAME ?? (() => { throw new Error('PACKAGE_NAME is not set in .env file'); })();
const MENTRAOS_API_KEY = process.env.MENTRAOS_API_KEY ?? (() => { throw new Error('MENTRAOS_API_KEY is not set in .env file'); })();
const ROBOFLOW_API_KEY = process.env.ROBOFLOW_API_KEY ?? (() => { throw new Error('ROBOFLOW_API_KEY is not set in .env file'); })();
const PORT = parseInt(process.env.PORT || '3000');

/**
 * Photo Taker App with webview functionality for displaying photos
 * Extends AppServer to provide photo taking and webview display capabilities
 */
class ExampleMentraOSApp extends AppServer {
  private photos: Map<string, StoredPhoto> = new Map(); // Store photos by userId
  private latestPhotoTimestamp: Map<string, number> = new Map(); // Track latest photo timestamp per user
  private isStreamingPhotos: Map<string, boolean> = new Map(); // Track if we are streaming photos for a user
  private nextPhotoTime: Map<string, number> = new Map(); // Track next photo time for a user
  private facesInPhoto: Map<string, FacePrediction[]> = new Map(); // Store face predictions by requestId

  constructor() {
    super({
      packageName: PACKAGE_NAME,
      apiKey: MENTRAOS_API_KEY,
      port: PORT,
    });
    this.setupWebviewRoutes();
  }


  /**
   * Handle new session creation and button press events
   */
  protected async onSession(session: AppSession, sessionId: string, userId: string): Promise<void> {
    // this gets called whenever a user launches the app
    this.logger.info(`Session started for user ${userId}`);

    // set the initial state of the user
    this.isStreamingPhotos.set(userId, false);
    this.nextPhotoTime.set(userId, Date.now());

    // this gets called whenever a user presses a button
    session.events.onButtonPress(async (button) => {
      this.logger.info(`Button pressed: ${button.buttonId}, type: ${button.pressType}`);

      if (button.pressType === 'long') {
        // the user held the button, so we toggle the streaming mode
        this.isStreamingPhotos.set(userId, !this.isStreamingPhotos.get(userId));
        this.logger.info(`Streaming photos for user ${userId} is now ${this.isStreamingPhotos.get(userId)}`);
        return;
      } else {
        session.layouts.showTextWall("Button pressed, about to take photo", {durationMs: 4000});
        // the user pressed the button, so we take a single photo
        try {
          // first, get the photo
          const photo = await session.camera.requestPhoto();
          this.logger.info(`Photo taken for user ${userId}, timestamp: ${photo.timestamp}`);
          this.cachePhoto(photo, userId);
        } catch (error) {
          this.logger.error(`Error taking photo: ${error}`);
        }
      }
    });

    // repeatedly check if we are in streaming mode and if we are ready to take another photo
    setInterval(async () => {
      if (this.isStreamingPhotos.get(userId) && Date.now() > (this.nextPhotoTime.get(userId) ?? 0)) {
        try {
          // set the next photos for 30 seconds from now, as a fallback if this fails
          this.nextPhotoTime.set(userId, Date.now() + 30000);

          // actually take the photo
          const photo = await session.camera.requestPhoto();

          // set the next photo time to now, since we are ready to take another photo
          this.nextPhotoTime.set(userId, Date.now());

          // cache the photo for display
          this.cachePhoto(photo, userId);
        } catch (error) {
          this.logger.error(`Error auto-taking photo: ${error}`);
        }
      }
    }, 1000);
  }

  protected async onStop(sessionId: string, userId: string, reason: string): Promise<void> {
    // clean up the user's state
    this.isStreamingPhotos.set(userId, false);
    this.nextPhotoTime.delete(userId);

    // Clean up old face detection data for this user
    this.cleanupOldFaceData(userId);

    this.logger.info(`Session stopped for user ${userId}, reason: ${reason}`);
  }

  /**
   * Clean up old face detection data to prevent memory leaks
   */
  private cleanupOldFaceData(userId: string): void {
    const currentPhoto = this.photos.get(userId);
    if (!currentPhoto) return;

    // Remove face data for all requestIds except the current one
    const entriesToDelete: string[] = [];
    for (const [requestId] of this.facesInPhoto) {
      if (requestId !== currentPhoto.requestId) {
        // Check if this requestId belongs to any current user's photo
        let isCurrentPhoto = false;
        for (const [, photo] of this.photos) {
          if (photo.requestId === requestId) {
            isCurrentPhoto = true;
            break;
          }
        }
        if (!isCurrentPhoto) {
          entriesToDelete.push(requestId);
        }
      }
    }

    // Delete old entries
    entriesToDelete.forEach(requestId => {
      this.facesInPhoto.delete(requestId);
      this.logger.debug(`Cleaned up face data for old requestId: ${requestId}`);
    });
  }

  /**
   * Cache a photo for display and detect faces using Roboflow API
   */
  private async cachePhoto(photo: PhotoData, userId: string) {
    // create a new stored photo object which includes the photo data and the user id
    const cachedPhoto: StoredPhoto = {
      requestId: photo.requestId,
      buffer: photo.buffer,
      timestamp: photo.timestamp,
      userId: userId,
      mimeType: photo.mimeType,
      filename: photo.filename,
      size: photo.size
    };

    // cache the photo for display
    this.photos.set(userId, cachedPhoto);
    // update the latest photo timestamp
    this.latestPhotoTimestamp.set(userId, cachedPhoto.timestamp.getTime());
    this.logger.info(`Photo cached for user ${userId}, timestamp: ${cachedPhoto.timestamp}`);

    // Clean up old face detection data periodically
    this.cleanupOldFaceData(userId);

    // detect faces using Roboflow API
    try {
      await this.detectFaces(photo.buffer, photo.requestId, userId);
    } catch (error) {
      this.logger.error(`Error detecting faces for user ${userId}, requestId ${photo.requestId}: ${error}`);
      // Store empty array if face detection fails
      this.facesInPhoto.set(photo.requestId, []);
    }
  }

    /**
   * Detect faces in a photo using the Roboflow API
   */
  private async detectFaces(imageBuffer: Buffer, requestId: string, userId: string): Promise<void> {
    try {
      // Convert buffer to base64
      const base64Image = imageBuffer.toString('base64');

      // Call Roboflow API
      const response = await axios({
        method: "POST",
        url: "https://serverless.roboflow.com/face-detection-mik1i/27",
        params: {
          api_key: ROBOFLOW_API_KEY
        },
        data: base64Image,
        headers: {
          "Content-Type": "application/x-www-form-urlencoded"
        }
      });

      const faceDetectionResult: FaceDetectionResponse = response.data;

      // Store the face predictions for this specific photo
      this.facesInPhoto.set(requestId, faceDetectionResult.predictions);

      this.logger.info(`Face detection completed for user ${userId}, requestId ${requestId}, found ${faceDetectionResult.predictions.length} faces`);
    } catch (error) {
      this.logger.error(`Roboflow API error for user ${userId}, requestId ${requestId}: ${error}`);
    }
  }


  /**
 * Set up webview routes for photo display functionality
 */
  private setupWebviewRoutes(): void {
    const app = this.getExpressApp();

    // API endpoint to get the latest photo for the authenticated user
    app.get('/api/latest-photo', (req: any, res: any) => {
      const userId = (req as AuthenticatedRequest).authUserId;

      if (!userId) {
        res.status(401).json({ error: 'Not authenticated' });
        return;
      }

      const photo = this.photos.get(userId);
      if (!photo) {
        res.status(404).json({ error: 'No photo available' });
        return;
      }

      res.json({
        requestId: photo.requestId,
        timestamp: photo.timestamp.getTime(),
        hasPhoto: true
      });
    });

    // API endpoint to get photo data
    app.get('/api/photo/:requestId', (req: any, res: any) => {
      const userId = (req as AuthenticatedRequest).authUserId;
      const requestId = req.params.requestId;

      if (!userId) {
        res.status(401).json({ error: 'Not authenticated' });
        return;
      }

      const photo = this.photos.get(userId);
      if (!photo || photo.requestId !== requestId) {
        res.status(404).json({ error: 'Photo not found' });
        return;
      }

      res.set({
        'Content-Type': photo.mimeType,
        'Cache-Control': 'no-cache'
      });
      res.send(photo.buffer);
    });

    // API endpoint to get face detection results for a specific photo
    app.get('/api/faces/:requestId', (req: any, res: any) => {
      const userId = (req as AuthenticatedRequest).authUserId;
      const requestId = req.params.requestId;

      if (!userId) {
        res.status(401).json({ error: 'Not authenticated' });
        return;
      }

      // Verify the photo belongs to this user
      const photo = this.photos.get(userId);
      if (!photo || photo.requestId !== requestId) {
        res.status(404).json({ error: 'Photo not found or not authorized' });
        return;
      }

      const faces = this.facesInPhoto.get(requestId);
      if (!faces) {
        res.status(404).json({ error: 'No face data available yet' });
        return;
      }

      res.json({
        faces: faces,
        count: faces.length,
        requestId: requestId
      });
    });

    // Main webview route - displays the photo viewer interface
    app.get('/webview', async (req: any, res: any) => {
      const userId = (req as AuthenticatedRequest).authUserId;

      if (!userId) {
        res.status(401).send(`
          <html>
            <head><title>Photo Viewer - Not Authenticated</title></head>
            <body style="font-family: Arial, sans-serif; text-align: center; padding: 50px;">
              <h1>Please open this page from the MentraOS app</h1>
            </body>
          </html>
        `);
        return;
      }

      const templatePath = path.join(process.cwd(), 'views', 'photo-viewer.ejs');
      const html = await ejs.renderFile(templatePath, {});
      res.send(html);
    });
  }
}



// Start the server
// DEV CONSOLE URL: https://console.mentra.glass/
// Get your webhook URL from ngrok (or whatever public URL you have)
const app = new ExampleMentraOSApp();

app.start().catch(console.error);