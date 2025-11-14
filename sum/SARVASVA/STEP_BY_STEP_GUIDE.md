# Step-by-Step Guide: Running and Testing the PDF Document Processor

## Prerequisites Check

Before starting, make sure you have:
- ✅ Python 3.8 or higher installed
- ✅ Internet connection (for downloading models on first run)
- ✅ A PDF document to test with

---

## Step 1: Install Dependencies

### Open Terminal/Command Prompt

**Windows:**
- Press `Win + R`, type `cmd` or `powershell`, press Enter

**Mac/Linux:**
- Open Terminal application

### Navigate to Project Directory

```bash
cd "C:\Users\incre\OneDrive\Desktop\SARVASVA"
```

### Install Required Packages

```bash
pip install -r requirements.txt
```

**Expected Output:**
```
Collecting PyMuPDF>=1.23.0
Collecting torch>=2.0.0
Collecting transformers>=4.30.0
...
Successfully installed ...
```

**⏱️ Time:** 5-10 minutes (depending on internet speed)

**✅ Check:** If you see "Successfully installed", you're good to go!

---

## Step 2: Prepare a Test PDF

You need a PDF document to test with. You can:
- Use any existing PDF document
- Create a simple test PDF with headings and subheadings
- Download a sample PDF from the internet

**Recommended:** Use a PDF that has:
- Clear headings (like "Introduction", "Methods", "Results")
- Some subheadings
- Regular content text

---

## Step 3: Launch the Web Frontend

### Run Streamlit

```bash
streamlit run app.py
```

**Expected Output:**
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

**✅ Check:** 
- Terminal shows the URLs above
- Browser should automatically open (if not, manually open http://localhost:8501)

**⚠️ Note:** The first time you run this, it may take a few minutes to download ML models.

---

## Step 4: Test the Application

### 4.1 Upload PDF

1. **In the browser**, look at the **left sidebar**
2. You'll see "📤 Upload PDF" section
3. Click **"Browse files"** or drag and drop your PDF
4. **✅ Check:** PDF filename appears in the sidebar

### 4.2 Process Document

1. After uploading, you'll see a **"🚀 Process Document"** button
2. Click the button
3. **✅ Check:** You should see:
   - A spinner saying "🔄 Processing document..."
   - Progress messages in the terminal
   - Success message: "✅ Document processed successfully!"

**Expected Terminal Output:**
```
📄 Step 1: Indexing document by page and line number...
   ✓ Indexed XXX lines

🤖 Step 2: Classifying lines (heading/subheading/content)...
   ✓ Classified XXX lines
   - Headings: X, Subheadings: X, Content: X

🏗️  Step 3: Building hierarchical structure...
   ✓ Built hierarchy with X headings

📋 Step 4: Creating content overview...
   ✓ Created content overview with X entries
```

### 4.3 View Statistics

After processing, you should see **4 metric cards** at the top:
- 📄 Total Headings
- 📑 Total Subheadings
- 📊 Total Sections
- 🔄 Process New Document button

**✅ Check:** Numbers appear in these cards (not all zeros)

### 4.4 Test Content Overview Tab

1. Click on the **"📋 Content Overview"** tab
2. **✅ Check:** You should see:
   - A list of headings
   - Each heading is expandable (click to expand)
   - Subheadings listed under each heading
   - Page and line numbers displayed

**What to look for:**
- Headings are numbered (1., 2., 3., etc.)
- Subheadings are nested (1.1, 1.2, etc.)
- Page numbers are shown
- Line ranges are displayed

### 4.5 Test Section Explorer Tab

1. Click on the **"🔍 Explore Sections"** tab
2. **✅ Check:** You should see:
   - A dropdown to select headings
   - Heading information displayed
   - Subheading dropdown (if subheadings exist)

3. **Select a heading** from the dropdown
4. **✅ Check:** Heading details appear below

5. **If subheadings exist:**
   - Select a subheading from the dropdown
   - Click **"📝 Generate Summary"** button
   - **✅ Check:** 
     - Spinner appears: "🔄 Generating summary..."
     - Summary appears in a styled box
     - Summary text is readable and relevant

**Expected Summary Output:**
- A box with the subheading name
- Summary text (2-5 sentences typically)
- Should make sense related to the subheading

### 4.6 Test Statistics Tab

1. Click on the **"📊 Statistics"** tab
2. **✅ Check:** You should see:
   - Multiple metric cards with numbers
   - A bar chart showing classification distribution
   - Download button for JSON results

3. **Click "📥 Download Processing Results (JSON)"**
4. **✅ Check:** 
   - Download button appears
   - Click it to download a JSON file
   - File should download to your Downloads folder

---

## Step 5: Verify All Features Work

### ✅ Feature Checklist

Go through each feature and verify:

- [ ] **Indexing:** PDF uploads and processes without errors
- [ ] **Classification:** Headings and subheadings are detected
- [ ] **Hierarchy:** Nested structure is built correctly
- [ ] **Content Overview:** Table of contents is displayed
- [ ] **Summarization:** Summaries are generated when clicking subheadings
- [ ] **Statistics:** Numbers and charts are displayed
- [ ] **Download:** JSON file can be downloaded

---

## Step 6: Test Command Line Interface (Optional)

### Open a New Terminal

Keep the Streamlit app running, open a **new terminal window**.

### Run Command Line Version

```bash
cd "C:\Users\incre\OneDrive\Desktop\SARVASVA"
python main.py your_document.pdf
```

**Replace `your_document.pdf` with your actual PDF filename**

**Expected Output:**
```
📄 Step 1: Indexing document by page and line number...
   ✓ Indexed XXX lines

🤖 Step 2: Classifying lines (heading/subheading/content)...
   ✓ Classified XXX lines
   - Headings: X, Subheadings: X, Content: X

🏗️  Step 3: Building hierarchical structure...
   ✓ Built hierarchy with X headings

📋 Step 4: Creating content overview...
   ✓ Created content overview with X entries

================================================================================
CONTENT OVERVIEW (Table of Contents)
================================================================================

1. [Your Heading]
   Page X, Line X
   Range: Page X-X, Lines X-X
   Subheadings (X):
      1.1 [Subheading]
      ...
```

**✅ Check:** 
- Processing completes without errors
- Content overview is printed
- File `processing_results.json` is created in the project folder

---

## Troubleshooting

### Problem: "ModuleNotFoundError" or "No module named 'streamlit'"

**Solution:**
```bash
pip install streamlit pandas
```

### Problem: "FileNotFoundError" when uploading PDF

**Solution:**
- Make sure the PDF file is not corrupted
- Try a different PDF file
- Check file permissions

### Problem: Processing takes too long or hangs

**Solution:**
- First run downloads models (can take 5-10 minutes)
- Check internet connection
- Wait for model downloads to complete
- Try with a smaller PDF first

### Problem: No headings detected

**Solution:**
- Your PDF might not have clear headings
- Try a PDF with obvious headings (like a research paper or book chapter)
- Check if headings are in uppercase or title case

### Problem: Summary generation fails

**Solution:**
- Check internet connection (models may need to download)
- Try a different subheading
- Check terminal for error messages

### Problem: Browser doesn't open automatically

**Solution:**
- Manually open browser
- Go to: `http://localhost:8501`
- Check if port 8501 is available

### Problem: "Port already in use"

**Solution:**
```bash
# Stop the current Streamlit app (Ctrl+C)
# Or use a different port:
streamlit run app.py --server.port 8502
```

---

## Success Indicators

### ✅ Everything is Working If:

1. **Upload works:** PDF appears in sidebar after upload
2. **Processing works:** Success message appears, statistics show numbers
3. **Overview works:** Headings and subheadings are listed
4. **Explorer works:** Can select headings and see details
5. **Summarization works:** Summary appears when clicking "Generate Summary"
6. **Statistics work:** Charts and numbers are displayed
7. **Download works:** JSON file downloads successfully

### ❌ Something is Wrong If:

1. Error messages appear in terminal
2. Browser shows error page
3. Processing never completes
4. All statistics show zero
5. No headings are detected
6. Summaries fail to generate

---

## Quick Test Summary

**Fastest way to verify it works:**

1. ✅ Run: `streamlit run app.py`
2. ✅ Upload a PDF
3. ✅ Click "Process Document"
4. ✅ See statistics with numbers > 0
5. ✅ Click "Explore Sections" tab
6. ✅ Select a subheading and generate summary
7. ✅ See summary appear

**If all 7 steps work → ✅ System is working correctly!**

---

## Next Steps After Verification

Once everything works:

1. Try with different PDFs
2. Experiment with different document types
3. Customize models if needed
4. Fine-tune classifier for better accuracy
5. Integrate into your workflow

---

## Need Help?

If something doesn't work:
1. Check the terminal for error messages
2. Verify all dependencies are installed
3. Try with a simpler PDF first
4. Check the troubleshooting section above
5. Review the error messages carefully


