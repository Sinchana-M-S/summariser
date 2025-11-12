import { useState } from 'react';
import { motion } from 'framer-motion';
import { Sparkles, Video, FileText, BookOpen } from 'lucide-react';
import api from '../../lib/api';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/Card';
import Button from '../../components/ui/Button';
import toast from 'react-hot-toast';

export default function StudentSummarizer() {
  const [courseId, setCourseId] = useState('');
  const [videoIndex, setVideoIndex] = useState('');
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(false);
  const [resourceText, setResourceText] = useState('');

  const handleSummarize = async (type) => {
    setLoading(true);
    setSummary('');
    
    try {
      let textToSummarize = '';
      
      if (type === 'course') {
        if (!courseId) {
          toast.error('Please enter a course ID');
          setLoading(false);
          return;
        }
        // For demo, use courseId as text. In production, fetch course content
        textToSummarize = `Course content for course ID: ${courseId}. This is a placeholder. In production, fetch actual course content.`;
      } else if (type === 'video') {
        if (!courseId || videoIndex === '') {
          toast.error('Please enter both course ID and video index');
          setLoading(false);
          return;
        }
        // For demo, use video info as text. In production, fetch video transcript
        textToSummarize = `Video transcript for course ${courseId}, video ${videoIndex}. This is a placeholder. In production, fetch actual video transcript.`;
      } else if (type === 'text' && resourceText) {
        textToSummarize = resourceText;
      } else {
        toast.error('Please provide content to summarize');
        setLoading(false);
        return;
      }

      const { data } = await api.post('/api/ai/summarize', {
        resource_text: textToSummarize,
        session_id: `summary_${Date.now()}`,
        language_code: 'en-IN'
      });

      setSummary(data.summary || 'Summary generated successfully!');
      toast.success('Summary created!');
    } catch (error) {
      console.error('Summarization error:', error);
      toast.error('Failed to summarize. Please check if AI service is running.');
      setSummary('Error: Could not generate summary. Please try again.');
    } finally {
      setLoading(false);
    }
  };


  return (
    <div className="min-h-screen bg-white dark:bg-black">
      <div className="container mx-auto px-4 py-20">
        <h1 className="mb-8 text-4xl font-bold text-black dark:text-white">AI Summarizer</h1>

        <div className="grid gap-8 lg:grid-cols-2">
          {/* Input Section */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Sparkles className="text-yellow-500 dark:text-yellow-400" size={24} />
                  <span>Summarize Course</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="mb-2 block text-sm text-gray-600 dark:text-gray-400">Course ID</label>
                  <input
                    type="text"
                    value={courseId}
                    onChange={(e) => setCourseId(e.target.value)}
                    placeholder="Enter course ID"
                    className="w-full rounded-lg border border-gray-300 dark:border-gray-800 bg-white dark:bg-gray-900 px-4 py-2 text-black dark:text-white outline-none focus:border-black dark:focus:border-white"
                  />
                </div>
                <Button
                  variant="primary"
                  className="w-full"
                  onClick={() => handleSummarize('course')}
                  disabled={loading}
                >
                  <BookOpen size={18} className="mr-2" />
                  {loading ? 'Summarizing...' : 'Summarize Course'}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <Video className="text-blue-500 dark:text-blue-400" size={24} />
                  <span>Summarize Video</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="mb-2 block text-sm text-gray-600 dark:text-gray-400">Course ID</label>
                  <input
                    type="text"
                    value={courseId}
                    onChange={(e) => setCourseId(e.target.value)}
                    placeholder="Enter course ID"
                    className="w-full rounded-lg border border-gray-300 dark:border-gray-800 bg-white dark:bg-gray-900 px-4 py-2 text-black dark:text-white outline-none focus:border-black dark:focus:border-white"
                  />
                </div>
                <div>
                  <label className="mb-2 block text-sm text-gray-600 dark:text-gray-400">Video Index</label>
                  <input
                    type="number"
                    value={videoIndex}
                    onChange={(e) => setVideoIndex(e.target.value)}
                    placeholder="Enter video index (0, 1, 2...)"
                    className="w-full rounded-lg border border-gray-300 dark:border-gray-800 bg-white dark:bg-gray-900 px-4 py-2 text-black dark:text-white outline-none focus:border-black dark:focus:border-white"
                  />
                </div>
                <Button
                  variant="primary"
                  className="w-full"
                  onClick={() => handleSummarize('video')}
                  disabled={loading}
                >
                  <Video size={18} className="mr-2" />
                  {loading ? 'Summarizing...' : 'Summarize Video'}
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="text-purple-500 dark:text-purple-400" size={24} />
                  <span>Summarize Text</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <label className="mb-2 block text-sm text-gray-600 dark:text-gray-400">Paste Text to Summarize</label>
                  <textarea
                    value={resourceText}
                    onChange={(e) => setResourceText(e.target.value)}
                    placeholder="Paste or type the content you want to summarize..."
                    rows={6}
                    className="w-full rounded-lg border border-gray-300 dark:border-gray-800 bg-white dark:bg-gray-900 px-4 py-2 text-black dark:text-white outline-none focus:border-black dark:focus:border-white resize-none"
                  />
                </div>
                <Button
                  variant="primary"
                  className="w-full"
                  onClick={() => handleSummarize('text')}
                  disabled={loading || !resourceText.trim()}
                >
                  <FileText size={18} className="mr-2" />
                  {loading ? 'Summarizing...' : 'Summarize Text'}
                </Button>
              </CardContent>
            </Card>
          </div>

          {/* Output Section */}
          <Card>
            <CardHeader>
                <CardTitle className="flex items-center space-x-2">
                  <FileText className="text-green-500 dark:text-green-400" size={24} />
                  <span>Summary</span>
                </CardTitle>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <div className="flex space-x-2">
                    <div className="h-3 w-3 animate-bounce rounded-full bg-black dark:bg-white" />
                    <div className="h-3 w-3 animate-bounce rounded-full bg-black dark:bg-white delay-75" />
                    <div className="h-3 w-3 animate-bounce rounded-full bg-black dark:bg-white delay-150" />
                  </div>
                </div>
              ) : summary ? (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="prose dark:prose-invert max-w-none"
                >
                  <p className="whitespace-pre-wrap text-gray-700 dark:text-gray-300">{summary}</p>
                </motion.div>
              ) : (
                <div className="py-12 text-center text-gray-600 dark:text-gray-400">
                  <Sparkles size={48} className="mx-auto mb-4 opacity-50" />
                  <p>Enter course or video details and click summarize</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}

